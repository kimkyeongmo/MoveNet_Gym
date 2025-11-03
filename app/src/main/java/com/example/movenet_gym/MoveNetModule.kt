package com.example.movenet_gym

import android.content.Intent
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod

class MoveNetModule(private val reactContext: ReactApplicationContext)
    : ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String {
        // JS에서 NativeModules.MoveNetModule 로 접근됨
        return "MoveNetModule"
    }

    @ReactMethod
    fun startMoveNetActivity() {
        // 카메라 인식용 MainActivity 실행
        val intent = Intent(reactContext, MainActivity::class.java)
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        reactContext.startActivity(intent)
    }
}
