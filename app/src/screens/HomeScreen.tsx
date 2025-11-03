import React from "react";
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from "react-native";

export default function HomeScreen({ navigate }: any) {
  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>AI 코치 🤖</Text>
      <Text style={styles.subtitle}>
        MoveNet 자세 교정과 LLM 플래너로 당신의 건강을 디자인하세요.
      </Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>실시간 자세 교정</Text>
        <Text style={styles.cardDesc}>카메라를 켜고 AI의 실시간 피드백을 받아보세요.</Text>

        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => navigate("coach")}
        >
          <Text style={styles.primaryButtonText}>자세 교정 시작하기</Text>
        </TouchableOpacity>

        <View style={styles.buttonRow}>
          <TouchableOpacity style={styles.miniButton}>
            <Text style={styles.miniButtonText}>스쿼트</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.miniButton}>
            <Text style={styles.miniButtonText}>런지</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.miniButton}>
            <Text style={styles.miniButtonText}>자유 자세</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>AI 플래너</Text>
        <Text style={styles.cardDesc}>운동 루틴과 식단 플랜을 생성하세요.</Text>

        <TouchableOpacity style={styles.secondaryButton}>
          <Text style={styles.secondaryButtonText}>루틴 생성하기</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0d1117", padding: 20 },
  title: { color: "white", fontSize: 28, fontWeight: "bold", marginBottom: 8 },
  subtitle: { color: "#9ca3af", fontSize: 14, marginBottom: 20 },
  card: {
    backgroundColor: "#161b22",
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
  },
  cardTitle: { color: "white", fontSize: 18, fontWeight: "bold", marginBottom: 8 },
  cardDesc: { color: "#aaa", marginBottom: 16 },
  primaryButton: {
    backgroundColor: "#3b82f6",
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
    marginBottom: 12,
  },
  primaryButtonText: { color: "white", fontWeight: "bold" },
  secondaryButton: {
    backgroundColor: "#e5e7eb",
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
  },
  secondaryButtonText: { color: "#111", fontWeight: "bold" },
  buttonRow: { flexDirection: "row", justifyContent: "space-between" },
  miniButton: {
    backgroundColor: "#1f2937",
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 12,
  },
  miniButtonText: { color: "#ccc", fontSize: 14 },
});
