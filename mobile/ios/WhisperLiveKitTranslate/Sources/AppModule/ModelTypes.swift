import Foundation

struct LanguageOption: Identifiable, Hashable {
    let code: String
    let name: String
    var id: String { code }
}

struct TranscriptLine: Identifiable, Hashable {
    let speaker: Int
    let text: String
    let translation: String?
    let detectedLanguage: String?
    let startTime: TimeInterval?
    let endTime: TimeInterval?

    /// Stable identifier built from the time range and speaker index so SwiftUI
    /// can diff incremental updates without view churn.
    var id: String {
        if let startTime, let endTime {
            // Use millisecond precision to avoid floating point noise in the key.
            let startKey = Int(startTime * 1_000)
            let endKey = Int(endTime * 1_000)
            return "speaker_\(speaker)_\(startKey)_\(endKey)"
        }
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return "speaker_\(speaker)_\(trimmedText)"
    }

    var hasTranslation: Bool {
        guard let translation else { return false }
        return !translation.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var speakerLabel: String {
        guard speaker >= 0 else { return "Narrator" }
        return "Speaker \(speaker + 1)"
    }

    var detectedLanguageName: String? {
        guard let detectedLanguage else { return nil }
        return LanguageCatalog.displayName(for: detectedLanguage)
    }

    var timestampLabel: String? {
        guard let startTime else { return nil }
        if let endTime {
            return "\(Self.format(time: startTime)) – \(Self.format(time: endTime))"
        }
        return Self.format(time: startTime)
    }

    private static func format(time: TimeInterval) -> String {
        let formatter = Self.intervalFormatter
        return formatter.string(from: time) ?? ""
    }

    private static let intervalFormatter: DateComponentsFormatter = {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute, .second]
        formatter.unitsStyle = .positional
        formatter.zeroFormattingBehavior = [.pad]
        return formatter
    }()
}

extension TranscriptLine {
    init(speaker: Int, text: String, translation: String?, detectedLanguage: String?, start: String?, end: String?) {
        self.speaker = speaker
        self.text = text
        self.translation = translation
        self.detectedLanguage = detectedLanguage
        self.startTime = Self.parse(time: start)
        self.endTime = Self.parse(time: end)
    }

    private static func parse(time: String?) -> TimeInterval? {
        guard let time, !time.isEmpty else { return nil }
        if let numeric = TimeInterval(time) {
            return numeric
        }

        let components = time.split(separator: ":").map(String.init)
        guard !components.isEmpty else { return nil }

        var multiplier: TimeInterval = 1
        var seconds: TimeInterval = 0

        for component in components.reversed() {
            guard let value = TimeInterval(component) else { return nil }
            seconds += value * multiplier
            multiplier *= 60
        }
        return seconds
    }
}
