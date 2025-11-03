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
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.movenet_gym.ui.theme.MoveNet_GymTheme
import org.tensorflow.lite.Interpreter
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.charset.Charset
import java.util.concurrent.Executors
import kotlin.math.acos
import kotlin.math.sqrt
import androidx.compose.ui.viewinterop.AndroidView

class MainActivity : ComponentActivity() {
    private lateinit var movenet: Interpreter
    private lateinit var classifier: Interpreter
    private lateinit var poseTemplates: Map<String, PoseTemplate> //  추가된 JSON 로드 변수
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var keypoints = mutableStateListOf<FloatArray>()

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        movenet = Interpreter(loadModelFile("movenet_lightning_fp16.tflite"))
        classifier = Interpreter(loadModelFile("exercise_classifier.tflite"))
        poseTemplates = loadPoseTemplates() //  pose_reference.json 불러오기 추가

        var resultText by mutableStateOf("📸 카메라 로드 중...")
        var warningText by mutableStateOf("")

        setContent {
            MoveNet_GymTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { padding ->
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(padding)
                    ) {
                        var showGuide by remember { mutableStateOf(true) }

                        LaunchedEffect(Unit) {
                            kotlinx.coroutines.delay(10000)
                            showGuide = false
                        }

                        CameraPreviewView { bitmap ->
                            analysisExecutor.execute {
                                val (res, kp, warn) = analyzeFrame(bitmap)
                                runOnUiThread {
                                    resultText = res
                                    warningText = warn
                                    keypoints.clear()
                                    keypoints.addAll(kp)
                                }
                            }
                        }

                        // 관절 시각화
                        SkeletonOverlay(keypoints = keypoints)
                        if (showGuide) HumanSilhouetteGuide()

                        if (warningText.isNotEmpty()) {
                            Box(
                                modifier = Modifier
                                    .align(Alignment.TopCenter)
                                    .padding(bottom = 8.dp)
                                    .background(
                                        color = Color.Black.copy(alpha = 0.5f),
                                        shape = androidx.compose.foundation.shape.RoundedCornerShape(16.dp)
                                    )
                                    .padding(horizontal = 20.dp, vertical = 0.dp)
                            ) {
                                Text(
                                    text = warningText,
                                    color = Color.White,
                                    textAlign = TextAlign.Center,
                                    style = MaterialTheme.typography.bodyLarge
                                )
                            }
                        }

                        Box(
                            modifier = Modifier
                                .align(Alignment.BottomCenter)
                                .padding(bottom = 24.dp)
                                .background(
                                    color = Color.Black.copy(alpha = 0.5f),
                                    shape = androidx.compose.foundation.shape.RoundedCornerShape(16.dp)
                                )
                                .padding(horizontal = 20.dp, vertical = 12.dp)
                                .padding(horizontal = 20.dp, vertical = 12.dp)
                        ) {
                            Text(
                                text = resultText,
                                color = Color.White,
                                textAlign = TextAlign.Center,
                                style = MaterialTheme.typography.bodyLarge
                            )
                        }

                    }
                }
            }
        }
    }

    //  자세 분석 함수 (기존 코드 유지 + JSON 비교 추가)
    private fun analyzeFrame(bitmap: Bitmap): Triple<String, List<FloatArray>, String> {
        return try {
            val matrix = android.graphics.Matrix().apply { preScale(-1f, 1f) }
            val mirrored = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            val inputBuffer = convertBitmapToByteBuffer(mirrored, 192, 192)

            val movenetOutput = Array(1) { Array(1) { Array(17) { FloatArray(3) } } }
            movenet.run(inputBuffer, movenetOutput)
            val keypoints = movenetOutput[0][0].toList()

            val avgScore = keypoints.map { it[2] }.average()
            val warning = if (avgScore < 0.3) "📍 전신이 모두 카메라에 보이게 서주세요" else ""

            val shoulder = keypoints[6]
            val elbow = keypoints[8]
            val wrist = keypoints[10]
            val hip = keypoints[12]
            val knee = keypoints[14]
            val ankle = keypoints[16]

            fun angle(p1: FloatArray, p2: FloatArray, p3: FloatArray): Float {
                val x1 = p1[1]; val y1 = p1[0]
                val x2 = p2[1]; val y2 = p2[0]
                val x3 = p3[1]; val y3 = p3[0]
                val v1x = x1 - x2; val v1y = y1 - y2
                val v2x = x3 - x2; val v2y = y3 - y2
                val dot = v1x * v2x + v1y * v2y
                val mag1 = sqrt(v1x * v1x + v1y * v1y)
                val mag2 = sqrt(v2x * v2x + v2y * v2y)
                val cos = (dot / (mag1 * mag2)).coerceIn(-1f, 1f)
                return Math.toDegrees(acos(cos.toDouble())).toFloat()
            }

            val inputAngles = floatArrayOf(
                1f,
                angle(hip, shoulder, elbow) / 180f,
                angle(shoulder, elbow, wrist) / 180f,
                angle(shoulder, hip, knee) / 180f,
                angle(hip, knee, ankle) / 180f,
                90f / 180f
            )

            val inputBuffer2 = ByteBuffer.allocateDirect(4 * inputAngles.size)
                .order(ByteOrder.nativeOrder())
            inputAngles.forEach { inputBuffer2.putFloat(it) }

            val output = Array(1) { FloatArray(5) }
            classifier.run(inputBuffer2, output)
            val prediction = output[0]
            val index = prediction.indices.maxByOrNull { prediction[it] } ?: -1
            val labels = listOf("Jumping Jacks", "Lunges", "Push Ups", "Sit Ups", "Squats")

            val result = if (index in labels.indices) "🏋️ 운동 인식: ${labels[index]}" else "❓ 인식 실패"

            //  JSON 피드백 비교 추가
            val currentExercise = if (index in labels.indices) labels[index] else "Unknown"
            val template = poseTemplates.entries.firstOrNull { it.key.startsWith(currentExercise) }?.value

            var feedbackText = ""
            template?.let {
                val currentAngles = mapOf(
                    "shoulder_angle" to angle(hip, shoulder, elbow),
                    "elbow_angle" to angle(shoulder, elbow, wrist),
                    "hip_angle" to angle(shoulder, hip, knee),
                    "knee_angle" to angle(hip, knee, ankle)
                )

                val mismatched = mutableListOf<String>()
                for ((name, refAngle) in it.angles) {
                    val current = currentAngles[name] ?: continue
                    val diff = kotlin.math.abs(current - refAngle)
                    if (diff > 15f) mismatched.add(name)
                }

                if (mismatched.isNotEmpty()) {
                    feedbackText = it.feedback.joinToString("\n")
                }
            }

            val finalWarning = if (feedbackText.isNotEmpty()) feedbackText else warning
            Triple(result, keypoints, finalWarning)
        } catch (e: Exception) {
            Triple("⚠️ 오류: ${e.message}", emptyList(), "")
        }
    }

    //  JSON 불러오기 함수 추가
    private fun loadPoseTemplates(): Map<String, PoseTemplate> {
        val jsonString = assets.open("pose_reference.json")
            .bufferedReader(Charset.forName("UTF-8"))
            .use { it.readText() }

        val jsonObject = JSONObject(jsonString)
        val templates = mutableMapOf<String, PoseTemplate>()

        val keys = jsonObject.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val obj = jsonObject.getJSONObject(key)
            val anglesObj = obj.getJSONObject("angles")
            val angles = mutableMapOf<String, Float>()
            anglesObj.keys().forEach {
                angles[it] = anglesObj.getDouble(it).toFloat()
            }

            val feedbackArr = obj.getJSONArray("feedback")
            val feedbackList = List(feedbackArr.length()) { feedbackArr.getString(it) }

            templates[key] = PoseTemplate(
                exercise = obj.getString("exercise"),
                phase = obj.getString("phase"),
                angles = angles,
                feedback = feedbackList
            )
        }
        return templates
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        val minEdge = minOf(bitmap.width, bitmap.height)
        val x0 = (bitmap.width - minEdge) / 2
        val y0 = (bitmap.height - minEdge) / 2
        val square = Bitmap.createBitmap(bitmap, x0, y0, minEdge, minEdge)
        val scaledBitmap = Bitmap.createScaledBitmap(square, width, height, true)

        val buffer = ByteBuffer.allocateDirect(4 * width * height * 3)
            .order(ByteOrder.nativeOrder())
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

    private fun loadModelFile(modelFile: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }
}

//  데이터 클래스 (추가)
data class PoseTemplate(
    val exercise: String,
    val phase: String,
    val angles: Map<String, Float>,
    val feedback: List<String>
)

//  관절 스켈레톤 오버레이
@Composable
fun SkeletonOverlay(keypoints: List<FloatArray>) {
    if (keypoints.isEmpty()) return
    Canvas(modifier = Modifier.fillMaxSize()) {
        val w = size.width
        val h = size.height

        val pairs = listOf(
            5 to 7, 7 to 9, 6 to 8, 8 to 10, // arms
            5 to 6, 11 to 12, // torso
            5 to 11, 6 to 12, // torso connection
            11 to 13, 13 to 15, 12 to 14, 14 to 16 // legs
        )

        for (pair in pairs) {
            val p1 = keypoints[pair.first]
            val p2 = keypoints[pair.second]
            val x1 = p1[1] * w
            val y1 = p1[0] * h
            val x2 = p2[1] * w
            val y2 = p2[0] * h
            drawLine(Color.Green, Offset(x1, y1), Offset(x2, y2), strokeWidth = 5f)
        }

        keypoints.forEach {
            drawCircle(Color.Yellow, radius = 8f, center = Offset(it[1] * w, it[0] * h))
        }
    }
}

//  사람 윤곽 가이드
@Composable
fun HumanSilhouetteGuide() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val w = size.width
            val h = size.height
            val cx = w / 2
            val topY = h * 0.15f
            val bottomY = h * 0.9f

            val path = androidx.compose.ui.graphics.Path().apply {
                moveTo(cx, topY)
                cubicTo(cx - w * 0.05f, topY + h * 0.05f, cx - w * 0.1f, topY + h * 0.15f, cx - w * 0.12f, topY + h * 0.25f)
                cubicTo(cx - w * 0.14f, topY + h * 0.35f, cx - w * 0.18f, topY + h * 0.45f, cx - w * 0.15f, topY + h * 0.55f)
                cubicTo(cx - w * 0.12f, topY + h * 0.65f, cx - w * 0.1f, bottomY - h * 0.05f, cx - w * 0.07f, bottomY)
                lineTo(cx + w * 0.07f, bottomY)
                cubicTo(cx + w * 0.1f, bottomY - h * 0.05f, cx + w * 0.12f, topY + h * 0.65f, cx + w * 0.15f, topY + h * 0.55f)
                cubicTo(cx + w * 0.18f, topY + h * 0.45f, cx + w * 0.14f, topY + h * 0.35f, cx + w * 0.12f, topY + h * 0.25f)
                cubicTo(cx + w * 0.1f, topY + h * 0.15f, cx + w * 0.05f, topY + h * 0.05f, cx, topY)
                close()
            }

            drawPath(path, Color.White, style = Stroke(width = 6f))
        }
//        Text(
//            text = "이 윤곽선 안에 전신이 다 보이게 서주세요",
//            color = Color.White,
//            style = MaterialTheme.typography.bodyLarge,
//            textAlign = TextAlign.Center,
//            modifier = Modifier
//                .align(Alignment.BottomCenter)
//                .padding(bottom = 30.dp)
//        )
    }
}

//  ImageProxy → Bitmap 변환
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

//  카메라 프리뷰 + 프레임 분석
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

            var lastAnalysis = 0L
            analyzer.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                val now = System.currentTimeMillis()
                if (now - lastAnalysis > 300) {
                    val bitmap = imageProxy.toBitmap()
                    if (bitmap != null) onFrame(bitmap)
                    lastAnalysis = now
                }
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
