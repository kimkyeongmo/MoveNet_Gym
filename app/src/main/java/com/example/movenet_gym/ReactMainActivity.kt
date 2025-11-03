package com.example.movenet_gym

import com.facebook.react.ReactActivity
import com.facebook.react.ReactActivityDelegate
import com.facebook.react.defaults.DefaultReactActivityDelegate
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint

class ReactMainActivity : ReactActivity() {

    override fun getMainComponentName(): String = "MoveNet_Gym"

    override fun createReactActivityDelegate(): ReactActivityDelegate {
        return DefaultReactActivityDelegate(
            this,
            mainComponentName,
            DefaultNewArchitectureEntryPoint.fabricEnabled, // Fabric 엔진 활성화
            DefaultNewArchitectureEntryPoint.concurrentReactEnabled // Concurrent React
        )
    }
}
