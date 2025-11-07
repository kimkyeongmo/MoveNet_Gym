package com.example.movenet_gym

import android.app.Application
import com.facebook.react.ReactApplication
import com.facebook.react.ReactNativeHost
import com.facebook.react.ReactPackage
import com.facebook.react.shell.MainReactPackage
import com.example.movenet_gym.BuildConfig
import com.example.movenet_gym.OpenAIPackage

class MainApplication : Application(), ReactApplication {

    private val mReactNativeHost = object : ReactNativeHost(this) {
        override fun getUseDeveloperSupport(): Boolean = BuildConfig.DEBUG

        override fun getPackages(): List<ReactPackage> {
            return listOf(
                MainReactPackage(),
                MoveNetPackage(),
                OpenAIPackage()
            )
        }

        override fun getJSMainModuleName(): String = "index"
    }

    override fun getReactNativeHost() = mReactNativeHost
}
