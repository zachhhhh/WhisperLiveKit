#if canImport(UIKit)
import SwiftUI

@available(iOS 16.0, *)
struct TranscriptHistoryView: View {
    let lines: [TranscriptLine]
    let interimText: String
    let showInterimPlaceholder: Bool

    private var hasInterimText: Bool {
        !interimText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    ForEach(lines) { line in
                        TranscriptRow(line: line)
                            .id(line.id)
                    }

                    if hasInterimText {
                        InterimRow(text: interimText)
                            .id("interim_row")
                    } else if showInterimPlaceholder && lines.isEmpty {
                        InterimPlaceholderRow()
                            .id("placeholder_row")
                    }
                }
                .padding(.vertical, 16)
            }
            .onChange(of: lines) { _ in
                scrollToBottom(proxy: proxy)
            }
            .onChange(of: interimText) { _ in
                if hasInterimText {
                    scrollToBottom(proxy: proxy)
                }
            }
        }
    }

    private func scrollToBottom(proxy: ScrollViewProxy) {
        DispatchQueue.main.async {
            if hasInterimText {
                proxy.scrollTo("interim_row", anchor: .bottom)
            } else if let lastId = lines.last?.id {
                proxy.scrollTo(lastId, anchor: .bottom)
            }
        }
    }
}

@available(iOS 16.0, *)
private struct TranscriptRow: View {
    let line: TranscriptLine

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Text(line.speakerLabel)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                if let language = line.detectedLanguageName {
                    Label(language, systemImage: "globe")
                        .labelStyle(.titleAndIcon)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let timestamp = line.timestampLabel {
                    Text(timestamp)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }

            Text(line.text)
                .font(.body.weight(.medium))
                .foregroundStyle(.primary)
                .frame(maxWidth: .infinity, alignment: .leading)

            if line.hasTranslation, let translation = line.translation?.trimmingCharacters(in: .whitespacesAndNewlines) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Translation")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(translation)
                        .font(.body)
                        .foregroundStyle(Color.blue)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            Divider()
        }
        .padding(.horizontal, 16)
    }
}

@available(iOS 16.0, *)
private struct InterimRow: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Listening…")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(text)
                .font(.body.weight(.semibold))
                .foregroundStyle(.primary)
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 8)
    }
}

@available(iOS 16.0, *)
private struct InterimPlaceholderRow: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Ready to transcribe")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("Tap record to stream live audio to WhisperLiveKit.")
                .font(.body)
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 8)
    }
}

#if DEBUG
struct TranscriptHistoryView_Previews: PreviewProvider {
    static var sampleLines: [TranscriptLine] = [
        TranscriptLine(
            speaker: 0,
            text: "Hello everyone, welcome to the meeting.",
            translation: "Hola a todos, bienvenidos a la reunión.",
            detectedLanguage: "en",
            start: "0.0",
            end: "3.2"
        ),
        TranscriptLine(
            speaker: 1,
            text: "Gracias, encantado de estar aquí.",
            translation: "Thank you, happy to be here.",
            detectedLanguage: "es",
            start: "3.5",
            end: "5.2"
        )
    ]

    static var previews: some View {
        TranscriptHistoryView(lines: sampleLines, interimText: "Processing latest audio…", showInterimPlaceholder: false)
    }
}
#endif
#endif
