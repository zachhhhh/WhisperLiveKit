package com.whisperlivekit.translate.model

data class TranscriptionLine(
    val speaker: Int,
    val text: String,
    val translation: String?,
    val start: String,
    val end: String,
    val detectedLanguage: String? = null,
)

data class TranslateUiState(
    val isStreaming: Boolean = false,
    val isConnected: Boolean = false,
    val transcriptLines: List<TranscriptionLine> = emptyList(),
    val bufferText: String = "",
    val translationPreview: String = "",
    val statusMessage: String? = null,
    val errorMessage: String? = null,
    val sourceLanguage: Language = LanguageCatalog.defaultSource,
    val targetLanguage: Language = LanguageCatalog.defaultTarget,
    val serverAddress: String = "ws://10.0.2.2:8000",
    val isPermissionMissing: Boolean = false,
)
