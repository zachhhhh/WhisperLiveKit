#if canImport(UIKit)
import SwiftUI

@available(iOS 16.0, *)
struct LanguagePickerView: View {
    let title: String
    let allowAuto: Bool
    @Binding var selection: LanguageOption
    let languages: [LanguageOption]

    @Environment(\.dismiss) private var dismiss
    @State private var query: String = ""

    private var filteredLanguages: [LanguageOption] {
        let sanitizedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !sanitizedQuery.isEmpty else {
            return languages
        }

        return languages.filter { option in
            option.name.localizedCaseInsensitiveContains(sanitizedQuery) || option.code.localizedCaseInsensitiveContains(sanitizedQuery)
        }
    }

    private var autoOption: LanguageOption? {
        guard allowAuto else { return nil }
        return languages.first { $0.code == LanguageCatalog.detectLanguageOption.code }
    }

    private var suggestionOptions: [LanguageOption] {
        guard query.isEmpty else { return [] }
        let suggestions = LanguageCatalog.popularLanguages
        return suggestions.filter { option in
            languages.contains(option) && option.code != LanguageCatalog.detectLanguageOption.code
        }
    }

    private var nonAutoLanguages: [LanguageOption] {
        filteredLanguages.filter { $0.code != LanguageCatalog.detectLanguageOption.code }
    }

    var body: some View {
        List {
            if let autoOption {
                Section("Detection") {
                    languageRow(for: autoOption)
                }
            }

            if !suggestionOptions.isEmpty {
                Section("Suggested") {
                    ForEach(suggestionOptions) { option in
                        languageRow(for: option)
                    }
                }
            }

            Section(query.isEmpty ? "All Languages" : "Matches") {
                ForEach(nonAutoLanguages) { option in
                    languageRow(for: option)
                }
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Cancel") { dismiss() }
            }
        }
        .searchable(text: $query, placement: .navigationBarDrawer(displayMode: .always))
    }

    private func languageRow(for option: LanguageOption) -> some View {
        Button {
            selection = option
            dismiss()
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(option.name)
                        .foregroundStyle(.primary)
                    Text(option.code)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if option == selection {
                    Image(systemName: "checkmark")
                        .foregroundStyle(Color.blue)
                }
            }
            .padding(.vertical, 4)
        }
    }
}

#if DEBUG
struct LanguagePickerView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationStack {
            LanguagePickerView(
                title: "Select source language",
                allowAuto: true,
                selection: .constant(LanguageCatalog.defaultTarget),
                languages: LanguageCatalog.availableLanguages
            )
        }
    }
}
#endif
#endif
