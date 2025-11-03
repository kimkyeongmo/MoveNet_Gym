/**
 * Metro configuration for React Native
 * https://github.com/facebook/metro
 *
 * @format
 */
 const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

 const defaultConfig = getDefaultConfig(__dirname);
 
 module.exports = mergeConfig(defaultConfig, {
   resolver: {
     assetExts: [...defaultConfig.resolver.assetExts, 'bin', 'tflite', 'json'],
   },
 });
 