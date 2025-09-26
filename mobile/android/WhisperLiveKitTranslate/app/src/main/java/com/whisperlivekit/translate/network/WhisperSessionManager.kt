package com.whisperlivekit.translate.network

import android.Manifest
import android.annotation.SuppressLint
import androidx.annotation.RequiresPermission
import com.whisperlivekit.translate.audio.AudioStreamer
import com.whisperlivekit.translate.model.Language
import com.whisperlivekit.translate.model.FrontDataMessage
import com.whisperlivekit.translate.model.ServerConfigMessage
import com.whisperlivekit.translate.model.SessionEvent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.contentOrNull
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import java.util.concurrent.atomic.AtomicBoolean

class WhisperSessionManager(
    private val client: OkHttpClient,
    private val json: Json,
    private val audioStreamer: AudioStreamer,
) {

    data class SessionConfig(
        val serverAddress: String,
        val sourceLanguage: Language,
        val targetLanguage: Language,
    )

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun openSession(config: SessionConfig): Flow<SessionEvent> = callbackFlow {
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val startedStreaming = AtomicBoolean(false)
        val request = try {
            buildRequest(config)
        } catch (t: Throwable) {
            trySend(SessionEvent.Failure(t))
            close(t)
            return@callbackFlow
        }

        val webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                // no-op
            }

            @SuppressLint("MissingPermission")
            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val element = json.parseToJsonElement(text)
                    if (element is JsonObject) {
                        val type = element["type"]?.jsonPrimitive?.contentOrNull
                        when (type) {
                            "config" -> {
                                val message = json.decodeFromJsonElement(ServerConfigMessage.serializer(), element)
                                if (message.useAudioWorklet != true) {
                                    trySend(
                                        SessionEvent.Failure(
                                            IllegalStateException("Server is not configured for PCM streaming. Restart with --pcm-input."),
                                        ),
                                    )
                                    return
                                }
                                trySend(SessionEvent.Config(message))

                                @SuppressLint("MissingPermission")
                                fun startStreamingAudio() {
                                    audioStreamer.start(
                                        scope = scope,
                                        onChunk = { chunk ->
                                            val sent = webSocket.send(chunk.toByteString())
                                            if (!sent) {
                                                trySend(SessionEvent.Failure(IllegalStateException("WebSocket send failed.")))
                                            }
                                        },
                                        onError = { throwable ->
                                            trySend(SessionEvent.Failure(throwable))
                                        },
                                    )
                                }

                                if (startedStreaming.compareAndSet(false, true)) {
                                    startStreamingAudio()
                                }
                            }

                            "ready_to_stop" -> {
                                trySend(SessionEvent.ReadyToStop)
                            }

                            else -> {
                                val payload = json.decodeFromJsonElement(FrontDataMessage.serializer(), element)
                                trySend(SessionEvent.FrontUpdate(payload))
                            }
                        }
                    }
                } catch (serialization: SerializationException) {
                    trySend(SessionEvent.Failure(serialization))
                } catch (throwable: Throwable) {
                    trySend(SessionEvent.Failure(throwable))
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                trySend(SessionEvent.Failure(t))
                close(t)
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                trySend(SessionEvent.Closed(reason = reason, code = code))
                webSocket.close(code, reason)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                trySend(SessionEvent.Closed(reason = reason, code = code))
            }
        })

        awaitClose {
            scope.launch {
                try {
                    audioStreamer.stopAndJoin()
                } finally {
                    runCatching { webSocket.send(ByteArray(0).toByteString()) }
                    webSocket.close(CLIENT_CLOSE_CODE, "Client requested stop")
                    scope.cancel()
                }
            }
        }
    }

    private fun buildRequest(config: SessionConfig): Request {
        val sanitized = config.serverAddress.trim().ifEmpty { DEFAULT_SERVER }
        val schemeAdjusted = when {
            sanitized.startsWith("ws://", ignoreCase = true) -> sanitized.replaceFirst("ws://", "http://", ignoreCase = true)
            sanitized.startsWith("wss://", ignoreCase = true) -> sanitized.replaceFirst("wss://", "https://", ignoreCase = true)
            sanitized.startsWith("http://", ignoreCase = true) -> sanitized
            sanitized.startsWith("https://", ignoreCase = true) -> sanitized
            else -> "http://$sanitized"
        }

        val httpUrl = schemeAdjusted.toHttpUrlOrNull()
            ?: throw IllegalArgumentException("Invalid server URL: ${config.serverAddress}")

        val builder = httpUrl.newBuilder().apply {
            val segments = httpUrl.pathSegments.filter { it.isNotEmpty() }
            encodedPath("/")
            segments.forEach { addPathSegment(it) }
            if (segments.lastOrNull() != "asr") {
                addPathSegment("asr")
            }
            if (config.sourceLanguage.code.isNotBlank()) {
                addQueryParameter("lan", config.sourceLanguage.code)
            }
            if (!config.targetLanguage.isAutoDetect && config.targetLanguage.code.isNotBlank()) {
                addQueryParameter("target_language", config.targetLanguage.code)
            }
        }

        val built = builder.build()
        val urlString = when {
            sanitized.startsWith("wss://", ignoreCase = true) || built.isHttps -> built.toString().replaceFirst("https://", "wss://")
            else -> built.toString().replaceFirst("http://", "ws://")
        }

        return Request.Builder().url(urlString).build()
    }

    companion object {
        private const val CLIENT_CLOSE_CODE = 1000
        private const val DEFAULT_SERVER = "ws://10.0.2.2:8000"
    }
}
