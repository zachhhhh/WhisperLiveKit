package com.whisperlivekit.translate.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.whisperlivekit.translate.model.Language
import com.whisperlivekit.translate.model.LanguageCatalog
import com.whisperlivekit.translate.model.TranslateUiState
import com.whisperlivekit.translate.ui.theme.WhisperLiveKitTheme
import kotlinx.coroutines.launch

@Composable
@OptIn(ExperimentalMaterial3Api::class)
fun TranslateApp() {
    WhisperLiveKitTheme {
        val viewModel: TranslateViewModel = viewModel()
        val state by viewModel.uiState.collectAsStateWithLifecycle()
        val waveform by viewModel.waveformEnergy.collectAsStateWithLifecycle(initialValue = emptyList())
        val context = LocalContext.current
        val snackbarHostState = remember { SnackbarHostState() }
        val coroutineScope = rememberCoroutineScope()
        val clipboard = LocalClipboardManager.current

        val permissionLauncher = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission(),
        ) { granted ->
            if (granted) {
                viewModel.toggleStreaming()
            } else {
                viewModel.onPermissionDenied()
            }
        }

        LaunchedEffect(state.errorMessage) {
            val error = state.errorMessage
            if (!error.isNullOrBlank()) {
                coroutineScope.launch {
                    snackbarHostState.showSnackbar(
                        message = error,
                        duration = SnackbarDuration.Long,
                    )
                    viewModel.clearError()
                }
            }
        }

        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("Whisper Live Translate") },
                    actions = {
                        val hasTranscript = state.transcriptLines.isNotEmpty() || state.bufferText.isNotBlank()
                        if (hasTranscript) {
                            IconButton(
                                onClick = {
                                    val combined = buildString {
                                        state.transcriptLines.forEachIndexed { index, line ->
                                            append("${index + 1}. ${line.text}\n")
                                            line.translation?.takeIf { it.isNotBlank() }?.let {
                                                append("→ $it\n")
                                            }
                                        }
                                        if (state.bufferText.isNotBlank()) {
                                            append(state.bufferText)
                                        }
                                    }
                                    clipboard.setText(AnnotatedString(combined.trim()))
                                    coroutineScope.launch {
                                        snackbarHostState.showSnackbar(
                                            message = "Transcript copied to clipboard",
                                            duration = SnackbarDuration.Short,
                                        )
                                    }
                                },
                            ) {
                                Icon(Icons.Default.ContentCopy, contentDescription = "Copy transcript")
                            }
                        }
                    },
                )
            },
            snackbarHost = { SnackbarHost(snackbarHostState) },
        ) { innerPadding ->
            TranslateContent(
                modifier = Modifier
                    .padding(innerPadding)
                    .fillMaxSize(),
                state = state,
                waveformEnergy = waveform,
                languages = LanguageCatalog.languages,
                onServerAddressChanged = viewModel::updateServerAddress,
                onSourceSelected = viewModel::selectSource,
                onTargetSelected = viewModel::selectTarget,
                onSwapLanguages = viewModel::swapLanguages,
                onDismissPermission = viewModel::acknowledgePermissionMessage,
                onToggleStreaming = {
                    val permission = ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.RECORD_AUDIO,
                    )
                    if (permission == PackageManager.PERMISSION_GRANTED) {
                        viewModel.toggleStreaming()
                    } else {
                        permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    }
                },
            )
        }
    }
}

@Composable
private fun TranslateContent(
    modifier: Modifier = Modifier,
    state: TranslateUiState,
    waveformEnergy: List<Float>,
    languages: List<Language>,
    onServerAddressChanged: (String) -> Unit,
    onSourceSelected: (Language) -> Unit,
    onTargetSelected: (Language) -> Unit,
    onSwapLanguages: () -> Unit,
    onDismissPermission: () -> Unit,
    onToggleStreaming: () -> Unit,
) {
    val scrollState = rememberScrollState()

    Column(
        modifier = modifier
            .background(MaterialTheme.colorScheme.surface)
            .verticalScroll(scrollState)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        AddressField(
            value = state.serverAddress,
            onValueChanged = onServerAddressChanged,
        )

        LanguageSelectionRow(
            sourceLanguage = state.sourceLanguage,
            targetLanguage = state.targetLanguage,
            languages = languages,
            onSourceSelected = onSourceSelected,
            onTargetSelected = onTargetSelected,
            onSwapLanguages = onSwapLanguages,
        )

        StatusBanner(state)

        if (state.isStreaming || waveformEnergy.isNotEmpty()) {
            WaveformVisualization(
                energyValues = waveformEnergy,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(96.dp),
            )
        }

        TranscriptCard(state)
        TranslationCard(state)

        RecordButton(
            isRecording = state.isStreaming,
            onToggle = onToggleStreaming,
        )

    }

    if (state.isPermissionMissing) {
        PermissionDialog(onDismiss = onDismissPermission)
    }
}

@Composable
private fun AddressField(
    value: String,
    onValueChanged: (String) -> Unit,
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChanged,
        modifier = Modifier.fillMaxWidth(),
        label = { Text("Server WebSocket URL") },
        supportingText = {
            Text("Example: ws://10.0.2.2:8000")
        },
        singleLine = true,
    )
}

@Composable
private fun StatusBanner(state: TranslateUiState) {
    val status = state.statusMessage
    if (status.isNullOrBlank()) return

    AssistChip(
        onClick = {},
        label = { Text(status) },
        leadingIcon = {
            Icon(Icons.AutoMirrored.Filled.VolumeUp, contentDescription = null)
        },
        colors = AssistChipDefaults.assistChipColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer,
            labelColor = MaterialTheme.colorScheme.onPrimaryContainer,
        ),
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun LanguageSelectionRow(
    sourceLanguage: Language,
    targetLanguage: Language,
    languages: List<Language>,
    onSourceSelected: (Language) -> Unit,
    onTargetSelected: (Language) -> Unit,
    onSwapLanguages: () -> Unit,
) {
    var pickSource by remember { mutableStateOf(false) }
    var pickTarget by remember { mutableStateOf(false) }

    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        LanguageButton(
            label = "From",
            language = sourceLanguage,
            onClick = { pickSource = true },
            modifier = Modifier.fillMaxWidth(0.45f),
        )

        OutlinedButton(
            onClick = onSwapLanguages,
            shape = CircleShape,
            colors = ButtonDefaults.outlinedButtonColors(),
        ) {
            Icon(Icons.Default.SwapHoriz, contentDescription = "Swap languages")
        }

        LanguageButton(
            label = "To",
            language = targetLanguage,
            onClick = { pickTarget = true },
            modifier = Modifier.fillMaxWidth(0.45f),
        )
    }

    if (pickSource) {
        LanguagePickerDialog(
            title = "Choose source language",
            languages = languages,
            onDismiss = { pickSource = false },
            onLanguageSelected = {
                onSourceSelected(it)
                pickSource = false
            },
        )
    }

    if (pickTarget) {
        LanguagePickerDialog(
            title = "Choose target language",
            languages = languages.filterNot { it.isAutoDetect },
            onDismiss = { pickTarget = false },
            onLanguageSelected = {
                onTargetSelected(it)
                pickTarget = false
            },
        )
    }
}

@Composable
private fun LanguageButton(
    label: String,
    language: Language,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier,
    ) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.height(4.dp))
        OutlinedButton(onClick = onClick, modifier = Modifier.fillMaxWidth()) {
            Text(
                text = language.displayName,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

@Composable
private fun TranscriptCard(state: TranslateUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Live transcript", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            state.transcriptLines.forEach { line ->
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = line.text,
                        style = MaterialTheme.typography.bodyLarge,
                    )
                    Text(
                        text = "${line.start} • ${line.end}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            if (state.bufferText.isNotBlank()) {
                Text(
                    text = state.bufferText,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun TranslationCard(state: TranslateUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                "Translation",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
                fontWeight = FontWeight.SemiBold,
            )
            if (state.translationPreview.isNotBlank()) {
                Text(
                    text = state.translationPreview,
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                )
            } else {
                Text(
                    text = "Translation will appear here",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f),
                )
            }
        }
    }
}

@Composable
private fun RecordButton(
    isRecording: Boolean,
    onToggle: () -> Unit,
) {
    val backgroundColor = if (isRecording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary

    Box(
        modifier = Modifier.fillMaxWidth(),
        contentAlignment = Alignment.Center,
    ) {
        OutlinedButton(
            onClick = onToggle,
            modifier = Modifier
                .size(96.dp),
            shape = CircleShape,
            colors = ButtonDefaults.outlinedButtonColors(
                containerColor = backgroundColor,
                contentColor = MaterialTheme.colorScheme.onPrimary,
            ),
            border = null,
        ) {
            val icon = if (isRecording) Icons.Default.Stop else Icons.Default.Mic
            Icon(
                imageVector = icon,
                contentDescription = if (isRecording) "Stop recording" else "Start recording",
                tint = MaterialTheme.colorScheme.onPrimary,
                modifier = Modifier.size(36.dp),
            )
        }
    }
}

@Composable
private fun PermissionDialog(onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Got it")
            }
        },
        icon = { Icon(Icons.Default.ErrorOutline, contentDescription = null) },
        title = { Text("Microphone permission needed") },
        text = { Text("Please grant microphone access to stream audio to the Whisper server.") },
    )
}

@Composable
private fun LanguagePickerDialog(
    title: String,
    languages: List<Language>,
    onDismiss: () -> Unit,
    onLanguageSelected: (Language) -> Unit,
) {
    var query by rememberSaveable { mutableStateOf("") }
    val filtered = remember(query, languages) {
        val trimmed = query.trim()
        if (trimmed.isEmpty()) languages else languages.filter {
            it.displayName.contains(trimmed, ignoreCase = true) ||
                it.code.contains(trimmed, ignoreCase = true)
        }
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
        title = { Text(title) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = query,
                    onValueChange = { query = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Search languages") },
                    singleLine = true,
                )
                LazyColumn(
                    modifier = Modifier.height(240.dp),
                ) {
                    items(filtered, key = { it.code }) { language ->
                        TextButton(
                            onClick = { onLanguageSelected(language) },
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Text(language.displayName)
                        }
                    }
                }
            }
        },
    )
}

@Composable
private fun WaveformVisualization(
    energyValues: List<Float>,
    modifier: Modifier = Modifier,
) {
    val primaryColor = MaterialTheme.colorScheme.primary
    val gridColor = primaryColor.copy(alpha = 0.15f)
    val emptyColor = primaryColor.copy(alpha = 0.2f)
    val fillColor = primaryColor.copy(alpha = 0.25f)

    Canvas(
        modifier = modifier
            .padding(vertical = 8.dp),
    ) {
        if (energyValues.isEmpty()) {
            drawLine(
                color = emptyColor,
                start = Offset.Zero,
                end = Offset(size.width, 0f),
                strokeWidth = 1f,
            )
            return@Canvas
        }

        val height = size.height
        val width = size.width
        val centerY = height / 2f

        repeat(5) { index ->
            val y = height * index / 4f
            drawLine(
                color = gridColor,
                start = Offset(0f, y),
                end = Offset(width, y),
                strokeWidth = 1f,
            )
        }

        val path = Path()
        val values = if (energyValues.size > MAX_ENERGY_POINTS) {
            energyValues.takeLast(MAX_ENERGY_POINTS)
        } else {
            energyValues
        }

        val spacing = if (values.size > 1) width / (values.size - 1) else width
        path.moveTo(0f, centerY)

        values.forEachIndexed { index, energy ->
            val x = index * spacing
            val clampedEnergy = energy.coerceIn(0f, 1f)
            val y = centerY - (clampedEnergy * height / 2f)
            path.lineTo(x, y)
        }

        values.asReversed().forEachIndexed { index, energy ->
            val x = (values.size - 1 - index) * spacing
            val clampedEnergy = energy.coerceIn(0f, 1f)
            val y = centerY + (clampedEnergy * height / 2f)
            path.lineTo(x, y)
        }

        path.close()

        drawPath(
            path = path,
            color = fillColor,
        )

        drawPath(
            path = path,
            color = primaryColor,
            style = Stroke(width = 2f),
        )
    }
}

private const val MAX_ENERGY_POINTS = 300
