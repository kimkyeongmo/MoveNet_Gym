package com.example.movenet_gym

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.movenet_gym.ui.theme.MoveNet_GymTheme
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import kotlin.math.acos
import kotlin.math.sqrt
import androidx.compose.ui.viewinterop.AndroidView


class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        var resultText by mutableStateOf("📸 카메라 로드 중...")

        setContent {
            MoveNet_GymTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { padding ->
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(padding),
                        contentAlignment = Alignment.Center
                    ) {
                        var showGuide by remember { mutableStateOf(true) }

                        // 5초 후 가이드 자동 제거
                        LaunchedEffect(Unit) {
                            kotlinx.coroutines.delay(5000)
                            showGuide = false
                        }

                        CameraPreviewView { bitmap ->
                            resultText = analyzeFrame(bitmap)
                        }

                        if (showGuide) {
                            GuideOverlay()
                        }

                        Column(
                            modifier = Modifier
                                .align(Alignment.BottomCenter)
                                .padding(bottom = 20.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(text = resultText, style = MaterialTheme.typography.bodyLarge)
                        }
                    }
                }
            }
        }
    }

    // 🔹 MoveNet + Classifier 분석
    private fun analyzeFrame(bitmap: Bitmap): String {
        return try {
            val movenet = Interpreter(loadModelFile("movenet_lightning_fp16.tflite"))
            val classifier = Interpreter(loadModelFile("exercise_classifier.tflite"))

            val inputBuffer = convertBitmapToByteBuffer(bitmap, 192, 192)
            val movenetOutput = Array(1) { Array(1) { Array(17) { FloatArray(3) } } }
            movenet.run(inputBuffer, movenetOutput)

            val keypoints = movenetOutput[0][0]
            val shoulder = keypoints[6]
            val elbow = keypoints[8]
            val wrist = keypoints[10]
            val hip = keypoints[12]
            val knee = keypoints[14]
            val ankle = keypoints[16]

            // 각도 계산 함수
            fun angle(p1: FloatArray, p2: FloatArray, p3: FloatArray): Float {
                val v1x = p1[0] - p2[0]
                val v1y = p1[1] - p2[1]
                val v2x = p3[0] - p2[0]
                val v2y = p3[1] - p2[1]
                val dot = v1x * v2x + v1y * v2y
                val mag1 = sqrt(v1x * v1x + v1y * v1y)
                val mag2 = sqrt(v2x * v2x + v2y * v2y)
                val cos = (dot / (mag1 * mag2)).coerceIn(-1f, 1f)
                return Math.toDegrees(acos(cos.toDouble())).toFloat()
            }

            // 각도 계산
            val inputAngles = floatArrayOf(
                1f,
                angle(hip, shoulder, elbow) / 180f,
                angle(shoulder, elbow, wrist) / 180f,
                angle(shoulder, hip, knee) / 180f,
                angle(hip, knee, ankle) / 180f,
                90f / 180f
            )

            val inputBuffer2 = ByteBuffer.allocateDirect(4 * inputAngles.size)
            inputBuffer2.order(ByteOrder.nativeOrder())
            inputAngles.forEach { inputBuffer2.putFloat(it) }

            val output = Array(1) { FloatArray(5) }
            classifier.run(inputBuffer2, output)

            val prediction = output[0]
            val index = prediction.indices.maxByOrNull { prediction[it] } ?: -1
            val labels = listOf("Jumping Jacks", "Lunges", "Push Ups", "Sit Ups", "Squats")

            if (index in labels.indices) {
                "🏋️ 운동 인식: ${labels[index]}"
            } else {
                "❓ 인식 실패"
            }
        } catch (e: Exception) {
            "⚠️ 오류: ${e.message}"
        }
    }

    // 🔹 Bitmap → ByteBuffer
    private fun convertBitmapToByteBuffer(bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val buffer = ByteBuffer.allocateDirect(4 * width * height * 3)
        buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(width * height)
        scaledBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        for (pixel in pixels) {
            val r = (pixel shr 16 and 0xFF) / 255f
            val g = (pixel shr 8 and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }
        buffer.rewind()
        return buffer
    }

    // 🔹 모델 로더
    private fun loadModelFile(modelFile: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }
}

// 🔹 카메라 프리뷰 + 분석
@Composable
fun CameraPreviewView(onFrame: (Bitmap) -> Unit) {
    val context = LocalContext.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    AndroidView(factory = { ctx ->
        val previewView = PreviewView(ctx)
        if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                ctx as ComponentActivity,
                arrayOf(Manifest.permission.CAMERA),
                10
            )
        }

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build()
            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analyzer.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                val bitmap = imageProxy.toBitmap()
                if (bitmap != null) onFrame(bitmap)
                imageProxy.close()
            }

            val selector = CameraSelector.DEFAULT_FRONT_CAMERA
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(ctx as ComponentActivity, selector, preview, analyzer)
            preview.setSurfaceProvider(previewView.surfaceProvider)
        }, ContextCompat.getMainExecutor(ctx))
        previewView
    }, modifier = Modifier.fillMaxSize())
}

// 🔹 가이드 오버레이 (5초간 표시)
@Composable
fun GuideOverlay() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val boxWidth = size.width * 0.6f
            val boxHeight = size.height * 0.7f
            val left = (size.width - boxWidth) / 2
            val top = (size.height - boxHeight) / 2

            drawRect(
                color = Color(0f, 1f, 0f, 0.5f),
                topLeft = Offset(left, top),
                size = Size(boxWidth, boxHeight),
                style = Stroke(width = 8f)
            )
        }

        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "📍 이 영역 안에 서주세요",
                color = Color.White,
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center
            )
        }
    }
}

// 🔹 ImageProxy → Bitmap 변환
fun ImageProxy.toBitmap(): Bitmap? {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer
    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()
    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)
    val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 90, out)
    val bytes = out.toByteArray()
    return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}
