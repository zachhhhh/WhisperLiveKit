package com.whisperlivekit.translate.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.annotation.RequiresPermission
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class AudioStreamer(
    private val sampleRateHz: Int = 16_000,
    private val chunkMillis: Int = 120,
) {
    private var audioRecord: AudioRecord? = null
    private var streamingJob: Job? = null
    private val energyWindow = ArrayDeque<Float>()
    private val _bufferEnergy = MutableStateFlow<List<Float>>(emptyList())
    val bufferEnergy: StateFlow<List<Float>> = _bufferEnergy

    @RequiresPermission(android.Manifest.permission.RECORD_AUDIO)
    fun start(
        scope: CoroutineScope,
        onChunk: (ByteArray) -> Unit,
        onError: (Throwable) -> Unit,
    ) {
        if (streamingJob != null) return

        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRateHz,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
            onError(IllegalStateException("Unable to determine buffer size for recording."))
            return
        }

        val bytesPerMillisecond = (sampleRateHz * BYTES_PER_SAMPLE) / 1_000
        val desiredBuffer = max(minBufferSize, bytesPerMillisecond * chunkMillis)

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRateHz,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            desiredBuffer,
        )

        val recorder = audioRecord
        if (recorder == null || recorder.state != AudioRecord.STATE_INITIALIZED) {
            onError(IllegalStateException("AudioRecord failed to initialize."))
            stop()
            return
        }

        recorder.startRecording()

        streamingJob = scope.launch(Dispatchers.IO) {
            val buffer = ByteArray(desiredBuffer)
            try {
                while (isActive) {
                    val read = recorder.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        val copy = buffer.copyOf(read)
                        onChunk(copy)
                        val energy = computeEnergy(copy, read)
                        updateEnergyWindow(energy)
                    }
                }
            } catch (t: Throwable) {
                onError(t)
            }
        }
    }

    suspend fun stopAndJoin() {
        streamingJob?.cancelAndJoin()
        streamingJob = null
        audioRecord?.run {
            try {
                stop()
            } catch (_: IllegalStateException) {
                // ignore stop failures when already stopped
            }
            release()
        }
        audioRecord = null
        resetEnergy()
    }

    fun stop() {
        streamingJob?.cancel()
        streamingJob = null
        audioRecord?.run {
            try {
                stop()
            } catch (_: IllegalStateException) {
            }
            release()
        }
        audioRecord = null
        resetEnergy()
    }

    companion object {
        private const val BYTES_PER_SAMPLE = 2
        private const val MAX_ENERGY_SAMPLES = 300
    }

    private fun computeEnergy(buffer: ByteArray, length: Int): Float {
        if (length <= 0) return 0f
        val sampleCount = length / BYTES_PER_SAMPLE
        if (sampleCount == 0) return 0f

        val byteBuffer = ByteBuffer.wrap(buffer, 0, length).order(ByteOrder.LITTLE_ENDIAN)
        var sumSquares = 0.0
        repeat(sampleCount) {
            val sample = byteBuffer.short / 32768.0
            sumSquares += sample * sample
        }
        val rms = sqrt(sumSquares / sampleCount)
        return min(1f, (rms * 5.0).toFloat())
    }

    private fun updateEnergyWindow(energy: Float) {
        energyWindow.addLast(energy)
        if (energyWindow.size > MAX_ENERGY_SAMPLES) {
            energyWindow.removeFirst()
        }
        _bufferEnergy.value = energyWindow.toList()
    }

    private fun resetEnergy() {
        energyWindow.clear()
        _bufferEnergy.value = emptyList()
    }
}
