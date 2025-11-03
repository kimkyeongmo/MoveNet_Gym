import React from "react";
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";

export default function PlanScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>AI 플래너 🍽️</Text>
      <Text style={styles.subtitle}>식단과 운동 루틴을 함께 설계하세요.</Text>

      <TouchableOpacity style={styles.button}>
        <Text style={styles.buttonText}>AI 루틴 생성하기</Text>
      </TouchableOpacity>

      <TouchableOpacity style={[styles.button, { backgroundColor: "#10b981" }]}>
        <Text style={styles.buttonText}>AI 식단 생성하기</Text>
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
    paddingHorizontal: 20,
  },
  title: { color: "white", fontSize: 26, fontWeight: "bold", marginBottom: 10 },
  subtitle: { color: "#9ca3af", marginBottom: 30 },
  button: {
    backgroundColor: "#3b82f6",
    paddingVertical: 14,
    paddingHorizontal: 40,
    borderRadius: 12,
    marginVertical: 10,
  },
  buttonText: { color: "white", fontWeight: "bold", fontSize: 16 },
});
