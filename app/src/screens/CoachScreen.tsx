import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { NativeModules } from "react-native";
const { MoveNetModule } = NativeModules;

export default function CoachScreen() {
  const startMoveNet = () => {
    MoveNetModule.startMoveNetActivity();
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>AI 자세 교정</Text>
      <Text style={styles.desc}>카메라를 통해 실시간으로 자세를 분석합니다.</Text>

      <TouchableOpacity style={styles.button} onPress={startMoveNet}>
        <Text style={styles.buttonText}>MoveNet 실행하기</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0d1117",
    justifyContent: "center",
    alignItems: "center",
  },
  title: { color: "white", fontSize: 24, fontWeight: "bold", marginBottom: 10 },
  desc: { color: "#9ca3af", textAlign: "center", marginBottom: 30 },
  button: {
    backgroundColor: "#3b82f6",
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 12,
  },
  buttonText: { color: "white", fontSize: 18, fontWeight: "bold" },
});
