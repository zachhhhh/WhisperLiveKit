import Foundation

/// Central catalogue of supported translation languages for the iOS client.
enum LanguageCatalog {
    static let detectLanguageOption = LanguageOption(code: "auto", name: "Detect Language")
    static let defaultTarget = LanguageOption(code: "en", name: "English")

    static let availableLanguages: [LanguageOption] = {
        let options = rawLanguages
            .map { LanguageOption(code: $0.code, name: $0.name) }
            .sorted { lhs, rhs in
                lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }
        return [detectLanguageOption] + options
    }()

    static let popularLanguages: [LanguageOption] = {
        let popularCodes: Set<String> = ["en", "es", "zh", "zh-Hant", "fr", "de", "ja", "ko", "pt", "hi"]
        return availableLanguages.filter { popularCodes.contains($0.code) }
    }()

    static func displayName(for code: String) -> String? {
        if code == detectLanguageOption.code {
            return detectLanguageOption.name
        }
        return languageLookup[code]
    }

    private static let rawLanguages: [(code: String, name: String)] = [
        ("af", "Afrikaans"),
        ("ak", "Akan"),
        ("am", "Amharic"),
        ("ar", "Arabic"),
        ("as", "Assamese"),
        ("ay", "Aymara"),
        ("az", "Azerbaijani"),
        ("be", "Belarusian"),
        ("bn", "Bengali"),
        ("bho", "Bhojpuri"),
        ("bg", "Bulgarian"),
        ("bm", "Bambara"),
        ("ca", "Catalan"),
        ("ceb", "Cebuano"),
        ("ny", "Chichewa"),
        ("zh", "Chinese (Simplified)"),
        ("zh-Hant", "Chinese (Traditional)"),
        ("co", "Corsican"),
        ("hr", "Croatian"),
        ("cs", "Czech"),
        ("da", "Danish"),
        ("dv", "Dhivehi"),
        ("doi", "Dogri"),
        ("nl", "Dutch"),
        ("dz", "Dzongkha"),
        ("en", "English"),
        ("eo", "Esperanto"),
        ("et", "Estonian"),
        ("ee", "Ewe"),
        ("fil", "Filipino"),
        ("fi", "Finnish"),
        ("fr", "French"),
        ("fy", "Frisian"),
        ("gl", "Galician"),
        ("ka", "Georgian"),
        ("de", "German"),
        ("el", "Greek"),
        ("gn", "Guarani"),
        ("gu", "Gujarati"),
        ("ht", "Haitian Creole"),
        ("ha", "Hausa"),
        ("haw", "Hawaiian"),
        ("he", "Hebrew"),
        ("hi", "Hindi"),
        ("hmn", "Hmong"),
        ("hu", "Hungarian"),
        ("is", "Icelandic"),
        ("ig", "Igbo"),
        ("ilo", "Ilocano"),
        ("id", "Indonesian"),
        ("ga", "Irish"),
        ("it", "Italian"),
        ("ja", "Japanese"),
        ("jw", "Javanese"),
        ("kn", "Kannada"),
        ("kk", "Kazakh"),
        ("km", "Khmer"),
        ("rw", "Kinyarwanda"),
        ("gom", "Konkani"),
        ("ko", "Korean"),
        ("kri", "Krio"),
        ("ku", "Kurdish"),
        ("ckb", "Kurdish (Sorani)"),
        ("ky", "Kyrgyz"),
        ("lo", "Lao"),
        ("la", "Latin"),
        ("lv", "Latvian"),
        ("lt", "Lithuanian"),
        ("lg", "Luganda"),
        ("lb", "Luxembourgish"),
        ("mk", "Macedonian"),
        ("mai", "Maithili"),
        ("mg", "Malagasy"),
        ("ms", "Malay"),
        ("ml", "Malayalam"),
        ("mt", "Maltese"),
        ("mi", "Maori"),
        ("mr", "Marathi"),
        ("mni-Mtei", "Meiteilon"),
        ("mn", "Mongolian"),
        ("my", "Myanmar"),
        ("ne", "Nepali"),
        ("no", "Norwegian"),
        ("or", "Odia"),
        ("om", "Oromo"),
        ("ps", "Pashto"),
        ("fa", "Persian"),
        ("pl", "Polish"),
        ("pt", "Portuguese"),
        ("pa", "Punjabi"),
        ("qu", "Quechua"),
        ("ro", "Romanian"),
        ("ru", "Russian"),
        ("sm", "Samoan"),
        ("sa", "Sanskrit"),
        ("gd", "Scots Gaelic"),
        ("nso", "Sepedi"),
        ("sr", "Serbian"),
        ("st", "Sesotho"),
        ("sn", "Shona"),
        ("sd", "Sindhi"),
        ("si", "Sinhala"),
        ("sk", "Slovak"),
        ("sl", "Slovenian"),
        ("so", "Somali"),
        ("es", "Spanish"),
        ("su", "Sundanese"),
        ("sw", "Swahili"),
        ("sv", "Swedish"),
        ("tg", "Tajik"),
        ("ta", "Tamil"),
        ("tt", "Tatar"),
        ("te", "Telugu"),
        ("th", "Thai"),
        ("ti", "Tigrinya"),
        ("ts", "Tsonga"),
        ("tr", "Turkish"),
        ("tk", "Turkmen"),
        ("tw", "Twi"),
        ("uk", "Ukrainian"),
        ("ur", "Urdu"),
        ("ug", "Uyghur"),
        ("uz", "Uzbek"),
        ("vi", "Vietnamese"),
        ("cy", "Welsh"),
        ("wo", "Wolof"),
        ("xh", "Xhosa"),
        ("yi", "Yiddish"),
        ("yo", "Yoruba"),
        ("zu", "Zulu")
    ]

    private static let languageLookup: [String: String] = {
        var dictionary = [String: String]()
        for entry in rawLanguages {
            dictionary[entry.code] = entry.name
        }
        return dictionary
    }()
}
