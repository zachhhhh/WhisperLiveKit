#if canImport(UIKit)
import SwiftUI

@available(iOS 16.0, *)
struct ContentView: View {
    @StateObject private var session = WhisperSession()
    @State private var host: String = "127.0.0.1"
    @State private var port: String = "8000"

    @State private var sourceLanguage: LanguageOption = LanguageCatalog.detectLanguageOption
    @State private var targetLanguage: LanguageOption = LanguageCatalog.defaultTarget

    @State private var showSourcePicker = false
    @State private var showTargetPicker = false
    @State private var showConnectionSheet = false

    var body: some View {
        NavigationStack {
            ZStack {
                Color(uiColor: UIColor.systemGroupedBackground)
                    .ignoresSafeArea()

                VStack(spacing: 24) {
                    languageSwitcher
                    conversationSection
                    micButton
                    Spacer(minLength: 12)
                    statusFooter
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 24)
            }
            .navigationTitle("Whisper Translate")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showConnectionSheet.toggle()
                    } label: {
                        Image(systemName: "gearshape")
                    }
                    .accessibilityLabel("Connection settings")
                }
            }
            .sheet(isPresented: $showSourcePicker) {
                NavigationStack {
                    LanguagePickerView(
                        title: "Select source language",
                        allowAuto: true,
                        selection: $sourceLanguage,
                        languages: session.languages
                    )
                }
            }
            .sheet(isPresented: $showTargetPicker) {
                NavigationStack {
                    LanguagePickerView(
                        title: "Select target language",
                        allowAuto: false,
                        selection: $targetLanguage,
                        languages: session.languages.filter { $0.code != "auto" }
                    )
                }
            }
            .sheet(isPresented: $showConnectionSheet) {
                NavigationStack {
                    ConnectionSettingsView(host: $host, port: $port)
                }
            }
            .onAppear {
                session.updateLanguages(source: sourceLanguage, target: targetLanguage)
            }
            .onChange(of: sourceLanguage) { newValue in
                session.updateLanguages(source: newValue, target: targetLanguage)
            }
            .onChange(of: targetLanguage) { newValue in
                session.updateLanguages(source: sourceLanguage, target: newValue)
            }
        }
    }

    private var languageSwitcher: some View {
        VStack(spacing: 16) {
            HStack(spacing: 12) {
                LanguageButton(
                    title: "From",
                    languageName: sourceLanguage.name,
                    action: { showSourcePicker = true }
                )

                Button(action: swapLanguages) {
                    Image(systemName: "arrow.left.arrow.right")
                        .font(.title3)
                        .padding(12)
                        .background(Circle().fill(Color.blue.opacity(0.15)))
                }
                .disabled(sourceLanguage.code == "auto")

                LanguageButton(
                    title: "To",
                    languageName: targetLanguage.name,
                    action: { showTargetPicker = true }
                )
            }

            Divider()
        }
    }

    private var conversationSection: some View {
        VStack(spacing: 16) {
            conversationHistoryCard
            translationCards
        }
    }

    private var conversationHistoryCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Conversation")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Live transcript")
                        .font(.title3.weight(.semibold))
                }
                Spacer()
                if !session.transcript.isEmpty {
                    Text("\(session.transcript.count) lines")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }

            TranscriptHistoryView(
                lines: session.transcript,
                interimText: session.interimText,
                showInterimPlaceholder: session.state == .idle
            )
            .frame(minHeight: 220)
            .frame(maxHeight: 320)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 24)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(Color.white)
        )
        .shadow(color: Color.black.opacity(0.05), radius: 12, x: 0, y: 6)
    }

    private var translationCards: some View {
        VStack(spacing: 16) {
            TranslationCard(
                title: "Original",
                text: displayedSourceText,
                placeholder: "Tap the microphone to start",
                alignment: .leading
            )

            TranslationCard(
                title: "Translation",
                text: displayedTranslatedText,
                placeholder: targetLanguagePlaceholder,
                alignment: .leading,
                accentColor: .blue
            )
        }
    }

    private var statusFooter: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(statusColor)
                .frame(width: 10, height: 10)

            Text(session.statusMessage.isEmpty ? "Idle" : session.statusMessage)
                .font(.footnote)
                .foregroundStyle(.secondary)

            Spacer()
        }
    }

    private var statusColor: Color {
        switch session.state {
        case .streaming: return .red
        case .connecting: return .orange
        case .error(_): return .red
        case .finishing: return .orange
        case .idle: return .gray
        }
    }

    private var displayedSourceText: String {
        if !session.interimText.isEmpty {
            return session.interimText
        }
        return session.transcript.last?.text ?? ""
    }

    private var displayedTranslatedText: String {
        if let translation = session.transcript.last?.translation, !translation.isEmpty {
            return translation
        }
        return ""
    }

    private var targetLanguagePlaceholder: String {
        if session.state == .streaming || !displayedSourceText.isEmpty {
            return "Awaiting translation…"
        }
        return "Translation appears here"
    }

    private var micButton: some View {
        VStack(spacing: 12) {
            Button(action: toggleSession) {
                ZStack {
                    Circle()
                        .fill(session.state == .streaming ? Color.red : Color.blue)
                        .frame(width: 96, height: 96)

                    Image(systemName: session.state == .streaming ? "stop.fill" : "mic.fill")
                        .font(.system(size: 40, weight: .semibold))
                        .foregroundColor(.white)
                }
                .shadow(color: Color.black.opacity(0.2), radius: 12, x: 0, y: 8)
            }

            if case .error(let message) = session.state {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
    }

    private func toggleSession() {
        switch session.state {
        case .idle, .error:
            guard let portValue = Int(port) else {
                session.statusMessage = "Invalid port"
                return
            }
            session.updateLanguages(source: sourceLanguage, target: targetLanguage)
            session.start(host: host, port: portValue)
        case .connecting, .streaming:
            session.stop()
        case .finishing:
            break
        }
    }

    private func swapLanguages() {
        guard sourceLanguage.code != "auto" else { return }
        let oldSource = sourceLanguage
        sourceLanguage = targetLanguage
        targetLanguage = oldSource
    }
}

@available(iOS 16.0, *)
private struct LanguageButton: View {
    let title: String
    let languageName: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 6) {
                Text(title.uppercased())
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack {
                    Text(languageName)
                        .font(.headline)
                        .foregroundStyle(.primary)
                    Spacer()
                    Image(systemName: "chevron.down")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.white)
                )
            }
        }
        .buttonStyle(.plain)
    }
}

@available(iOS 16.0, *)
private struct TranslationCard: View {
    enum Alignment {
        case leading
    }

    let title: String
    let text: String
    let placeholder: String
    let alignment: Alignment
    var accentColor: Color = .primary

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title.uppercased())
                .font(.caption)
                .foregroundStyle(.secondary)

            if text.isEmpty {
                Text(placeholder)
                    .font(.title3)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Text(text)
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(accentColor)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .animation(.easeInOut(duration: 0.2), value: text)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 24)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(Color.white)
        )
        .shadow(color: Color.black.opacity(0.05), radius: 12, x: 0, y: 6)
    }
}

#if DEBUG
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
#endif
#endif
