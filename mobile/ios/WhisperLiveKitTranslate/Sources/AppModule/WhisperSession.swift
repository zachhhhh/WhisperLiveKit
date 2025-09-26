#if canImport(UIKit)
import Foundation
import Combine
import AVFoundation

private struct ServerLine: Decodable {
    let speaker: Int
    let text: String
    let translation: String?
    let detected_language: String?
    let start: String
    let end: String
}

private struct ServerMessage: Decodable {
    let type: String?
    let status: String?
    let error: String?
    let lines: [ServerLine]?
    let buffer_transcription: String?
    let buffer_diarization: String?
    let remaining_time_transcription: Double?
    let remaining_time_diarization: Double?
    let useAudioWorklet: Bool?
    let source_language: String?
    let target_language: String?
}

enum SessionState: Equatable {
    case idle
    case connecting
    case streaming
    case finishing
    case error(String)
}

@MainActor
@available(iOS 16.0, *)
final class WhisperSession: NSObject, ObservableObject {
    @Published var state: SessionState = .idle
    @Published var transcript: [TranscriptLine] = []
    @Published var interimText: String = ""
    @Published var statusMessage: String = ""
    @Published var usePCMInput: Bool = true

    @Published private(set) var languages: [LanguageOption] = LanguageCatalog.availableLanguages

    private var webSocket: URLSessionWebSocketTask?
    private let urlSession: URLSession
    private var cancellables = Set<AnyCancellable>()

    private let audioCapture = AudioCapture()
    private let audioQueue = DispatchQueue(label: "com.whisperlivekit.audio")

    private var sourceLanguage: LanguageOption = LanguageCatalog.detectLanguageOption
    private var targetLanguage: LanguageOption = LanguageCatalog.defaultTarget
    private var serverURL: URL?
    private var awaitingConfig = false
    private var hasSentStopMarker = false

    override init() {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 60
        urlSession = URLSession(configuration: configuration)
        super.init()
    }

    func updateLanguages(source: LanguageOption, target: LanguageOption) {
        sourceLanguage = normalizedLanguage(for: source, fallback: LanguageCatalog.detectLanguageOption)
        targetLanguage = normalizedLanguage(for: target, fallback: LanguageCatalog.defaultTarget)
    }

    func start(host: String, port: Int) {
        switch state {
        case .idle, .error(_):
            break
        default:
            return
        }
        guard webSocket == nil else { return }
        guard let url = buildWebSocketURL(host: host, port: port) else {
            state = .error("Invalid server URL")
            return
        }
        serverURL = url
        transcript = []
        interimText = ""
        statusMessage = "Connecting…"
        awaitingConfig = true
        hasSentStopMarker = false
        state = .connecting

        let task = urlSession.webSocketTask(with: url)
        webSocket = task
        listenForMessages()
        task.resume()
    }

    func stop() {
        guard webSocket != nil else { return }
        state = .finishing
        sendStopMarker()
        audioQueue.async { [weak self] in
            guard let self else { return }
            self.audioCapture.stop()
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
        statusMessage = "Finishing session…"
    }

    private func buildWebSocketURL(host: String, port: Int) -> URL? {
        var components = URLComponents()
        components.scheme = "ws"
        components.host = host
        components.port = port
        components.path = "/asr"
        components.queryItems = [
            URLQueryItem(name: "source", value: sourceLanguage.code),
            URLQueryItem(name: "target", value: targetLanguage.code)
        ]
        return components.url
    }

    private func listenForMessages() {
        webSocket?.receive { [weak self] result in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.handleReceive(result)
            }
        }
    }

    private func handleReceive(_ result: Result<URLSessionWebSocketTask.Message, Error>) {
        switch result {
        case .success(let message):
            handle(message: message)
            listenForMessages()
        case .failure(let error):
            statusMessage = "Connection error: \(error.localizedDescription)"
            state = .error("WebSocket error")
            cleanupConnection()
        }
    }

    private func handle(message: URLSessionWebSocketTask.Message) {
        var data: Data?
        switch message {
        case .data(let d):
            data = d
        case .string(let text):
            data = text.data(using: .utf8)
        @unknown default:
            break
        }

        guard let data else { return }
        let decoder = JSONDecoder()
        if let serverMessage = try? decoder.decode(ServerMessage.self, from: data) {
            switch serverMessage.type {
            case "config":
                usePCMInput = serverMessage.useAudioWorklet ?? usePCMInput
                statusMessage = "Connected"
                if awaitingConfig {
                    awaitingConfig = false
                    Task { await startAudioCapture() }
                }
            case "ready_to_stop":
                statusMessage = "Server finished processing"
                state = .idle
                cleanupConnection()
            default:
                apply(serverMessage)
            }
        }
    }

    private func apply(_ message: ServerMessage) {
        if let error = message.error {
            state = .error(error)
            statusMessage = error
            return
        }

        if let lines = message.lines {
            transcript = lines.map {
                TranscriptLine(
                    speaker: $0.speaker,
                    text: $0.text,
                    translation: $0.translation,
                    detectedLanguage: $0.detected_language,
                    start: $0.start,
                    end: $0.end
                )
            }
        }

        interimText = message.buffer_transcription ?? ""

        if let status = message.status {
            switch status {
            case "no_audio_detected":
                statusMessage = "No speech detected"
            default:
                statusMessage = "Listening"
            }
        }

        if state == .connecting {
            state = .streaming
        }
    }

    private func startAudioCapture() async {
        do {
            try configureAudioSession()
            try audioCapture.start { [weak self] samples in
                guard let self else { return }
                self.audioQueue.async {
                    self.stream(samples: samples)
                }
            }
            statusMessage = "Recording"
            state = .streaming
        } catch {
            statusMessage = "Audio error: \(error.localizedDescription)"
            state = .error(error.localizedDescription)
            cleanupConnection()
        }
    }

    private func configureAudioSession() throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        #endif
    }

    private func stream(samples: [Float]) {
        guard !samples.isEmpty else { return }
        let duration = Float(samples.count) / Float(AudioCapture.sampleRate)
        let energyHistory = audioCapture.relativeEnergy
        let hasHistory = energyHistory.count >= 10
        if hasHistory && !AudioCapture.isVoiceDetected(in: energyHistory, nextBufferInSeconds: duration, silenceThreshold: 0.45) {
            return
        }

        var int16Samples = [Int16](repeating: 0, count: samples.count)
        for (index, sample) in samples.enumerated() {
            let clamped = max(-1.0, min(1.0, sample))
            int16Samples[index] = Int16(clamped * Float(Int16.max))
        }

        let data = int16Samples.withUnsafeBytes { Data($0) }
        webSocket?.send(.data(data)) { [weak self] error in
            guard let error else { return }
            Task { @MainActor [weak self] in
                self?.statusMessage = "Send error: \(error.localizedDescription)"
            }
        }
    }

    private func sendStopMarker() {
        guard !hasSentStopMarker else { return }
        hasSentStopMarker = true
        webSocket?.send(.data(Data())) { [weak self] error in
            guard let error else { return }
            Task { @MainActor [weak self] in
                self?.statusMessage = "Stop signal error: \(error.localizedDescription)"
            }
        }
    }

    private func cleanupConnection() {
        audioQueue.async { [weak self] in
            self?.audioCapture.stop()
        }
        webSocket?.cancel(with: .goingAway, reason: nil)
        webSocket = nil
        awaitingConfig = false
        hasSentStopMarker = false
    }
}

private extension WhisperSession {
    private func normalizedLanguage(for option: LanguageOption, fallback: LanguageOption) -> LanguageOption {
        if let match = languages.first(where: { $0.code == option.code }) {
            return match
        }
        return fallback
    }
}
#endif
