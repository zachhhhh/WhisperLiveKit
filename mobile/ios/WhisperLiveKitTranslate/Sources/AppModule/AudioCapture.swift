#if canImport(UIKit)
import AVFoundation
import Accelerate

/// Lightweight audio capture pipeline adapted from WhisperKit's AudioProcessor (MIT License © 2024 Argmax Inc.).
/// Converts microphone input to 16 kHz mono Float32 buffers and tracks energy for basic VAD heuristics.
final class AudioCapture: NSObject {
    static let sampleRate: Double = 16_000
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private var desiredFormat: AVAudioFormat?
    private var callback: (([Float]) -> Void)?
    private var isTapInstalled = false

    private let energyQueue = DispatchQueue(label: "AudioCapture.energy", qos: .userInitiated)
    private var energyHistory: [(relative: Float, average: Float)] = []
    var relativeEnergyWindow: Int = 20

    var relativeEnergy: [Float] {
        energyQueue.sync { energyHistory.map { $0.relative } }
    }

    func start(callback: @escaping ([Float]) -> Void) throws {
        self.callback = callback
        energyQueue.sync { energyHistory.removeAll(keepingCapacity: true) }

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        desiredFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Self.sampleRate, channels: 1, interleaved: false)

        guard let desiredFormat, let converter = AVAudioConverter(from: inputFormat, to: desiredFormat) else {
            throw NSError(domain: "AudioCapture", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to create audio converter"])
        }
        self.converter = converter

        inputNode.removeTap(onBus: 0)
        let bufferSize = AVAudioFrameCount(Self.sampleRate * 0.1) // 100 ms windows
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] buffer, _ in
            self?.handleIncomingBuffer(buffer)
        }
        isTapInstalled = true

        engine.prepare()
        try engine.start()
    }

    func stop() {
        if isTapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            isTapInstalled = false
        }
        engine.stop()
        callback = nil
        converter = nil
    }

    private func handleIncomingBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let desiredFormat, let converter else { return }

        do {
            let monoBuffer = try convert(buffer: buffer, with: converter, targetFormat: desiredFormat)
            let samples = AudioCapture.convertBufferToArray(buffer: monoBuffer)
            guard !samples.isEmpty else { return }
            updateEnergy(with: samples)
            callback?(samples)
        } catch {
            #if DEBUG
            print("AudioCapture conversion error: \(error.localizedDescription)")
            #endif
        }
    }

    private func updateEnergy(with samples: [Float]) {
        energyQueue.sync {
            let stats = AudioCapture.calculateEnergy(of: samples)
            let baseline = energyHistory.suffix(relativeEnergyWindow).map { $0.average }.min()
            let relative = AudioCapture.calculateRelativeEnergy(of: samples, relativeTo: baseline)
            energyHistory.append((relative: relative, average: stats.avg))
            let maxEntries = max(40, relativeEnergyWindow * 4)
            if energyHistory.count > maxEntries {
                energyHistory.removeFirst(energyHistory.count - maxEntries)
            }
        }
    }

    private func convert(buffer: AVAudioPCMBuffer, with converter: AVAudioConverter, targetFormat: AVAudioFormat) throws -> AVAudioPCMBuffer {
        if buffer.format.sampleRate == targetFormat.sampleRate && buffer.format.channelCount == targetFormat.channelCount {
            return buffer
        }

        let ratio = targetFormat.sampleRate / buffer.format.sampleRate
        let capacity = max(1, AVAudioFrameCount(Double(buffer.frameLength) * ratio))
        guard let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
            throw NSError(domain: "AudioCapture", code: -2, userInfo: [NSLocalizedDescriptionKey: "Unable to allocate output buffer"])
        }

        var conversionError: NSError?
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }

        let status = converter.convert(to: output, error: &conversionError, withInputFrom: inputBlock)
        if let conversionError {
            throw conversionError
        }
        #if os(macOS) || os(iOS)
        switch status {
        case .haveData, .inputRanDry:
            break
        case .endOfStream:
            break
        case .error:
            throw NSError(domain: "AudioCapture", code: -3, userInfo: [NSLocalizedDescriptionKey: "Audio conversion failed"])
        @unknown default:
            break
        }
        #else
        if !status {
            throw NSError(domain: "AudioCapture", code: -3, userInfo: [NSLocalizedDescriptionKey: "Audio conversion failed"])
        }
        #endif
        return output
    }
}

#endif

#if canImport(UIKit)
extension AudioCapture {
    static func isVoiceDetected(in relativeEnergy: [Float], nextBufferInSeconds: Float, silenceThreshold: Float) -> Bool {
        let energyValuesToConsider = max(0, Int(nextBufferInSeconds / 0.1))
        let nextBufferEnergies = relativeEnergy.suffix(energyValuesToConsider)
        let numberOfValuesToCheck = max(10, nextBufferEnergies.count - 10)
        return nextBufferEnergies.prefix(numberOfValuesToCheck).contains { $0 > silenceThreshold }
    }

    static func calculateAverageEnergy(of signal: [Float]) -> Float {
        var rmsEnergy: Float = 0.0
        vDSP_rmsqv(signal, 1, &rmsEnergy, vDSP_Length(signal.count))
        return rmsEnergy
    }

    static func calculateEnergy(of signal: [Float]) -> (avg: Float, max: Float, min: Float) {
        var rmsEnergy: Float = 0.0
        var minEnergy: Float = 0.0
        var maxEnergy: Float = 0.0
        vDSP_rmsqv(signal, 1, &rmsEnergy, vDSP_Length(signal.count))
        vDSP_maxmgv(signal, 1, &maxEnergy, vDSP_Length(signal.count))
        vDSP_minmgv(signal, 1, &minEnergy, vDSP_Length(signal.count))
        return (rmsEnergy, maxEnergy, minEnergy)
    }

    static func calculateRelativeEnergy(of signal: [Float], relativeTo reference: Float?) -> Float {
        let signalEnergy = calculateAverageEnergy(of: signal)
        let referenceEnergy = max(1e-8, reference ?? 1e-3)
        let dbEnergy = 20 * log10(signalEnergy)
        let refEnergy = 20 * log10(referenceEnergy)
        let normalizedEnergy = (dbEnergy - refEnergy) / (0 - refEnergy)
        return max(0, min(normalizedEnergy, 1))
    }

    static func convertBufferToArray(buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else { return [] }
        let frameLength = Int(buffer.frameLength)
        let pointer = channelData[0]
        return Array(UnsafeBufferPointer(start: pointer, count: frameLength))
    }
}
#endif
