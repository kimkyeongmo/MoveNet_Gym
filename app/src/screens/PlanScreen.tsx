import React, { useState } from "react";
import {
  View, Text, StyleSheet, TextInput, TouchableOpacity, ScrollView, NativeModules
} from "react-native";




export default function PlanScreen() {
  const [age, setAge] = useState("");
  const [height, setHeight] = useState("");
  const [weight, setWeight] = useState("");
  const [muscle, setMuscle] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {

    const { OpenAIModule } = NativeModules;
    
    const userProfile = {
      age: Number(age),
      height: Number(height),
      weight: Number(weight),
      muscleMass: muscle ? Number(muscle) : null
    };

    setLoading(true);
    try {
      const response = await OpenAIModule.generateRoutineAndDiet(JSON.stringify(userProfile));
      console.log("NativeModules:", NativeModules);
      setResult(response);
    } catch (e) {
      setResult(`⚠️ 오류: ${e}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>AI 플래너 🍽️</Text>
      <Text style={styles.subtitle}>당신의 신체정보를 입력하고 맞춤 루틴을 받아보세요</Text>

      <TextInput placeholder="나이 (예: 25)" placeholderTextColor="#FFFFFF" value={age} onChangeText={setAge} style={styles.input} keyboardType="numeric" />
      <TextInput placeholder="키 (cm)" placeholderTextColor="#FFFFFF" value={height} onChangeText={setHeight} style={styles.input} keyboardType="numeric" />
      <TextInput placeholder="몸무게 (kg)" placeholderTextColor="#FFFFFF" value={weight} onChangeText={setWeight} style={styles.input} keyboardType="numeric" />
      <TextInput placeholder="근육량 (선택, kg)" placeholderTextColor="#FFFFFF" value={muscle} onChangeText={setMuscle} style={styles.input} keyboardType="numeric" />

      <TouchableOpacity style={styles.button} onPress={handleGenerate}>
        <Text style={styles.buttonText}>AI 맞춤 플랜 생성하기</Text>
      </TouchableOpacity>

      {loading ? <Text style={styles.loading}>AI가 분석 중입니다...</Text> : null}
      {result ? <Text style={styles.result}>{result}</Text> : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flexGrow: 1, backgroundColor: "#0d1117", padding: 20, alignItems: "center" },
  title: { color: "white", fontSize: 26, fontWeight: "bold", marginBottom: 10 },
  subtitle: { color: "#9ca3af", marginBottom: 20 },
  input: {
    backgroundColor: "#161b22",
    color: "white",
    width: "90%",
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 10,
    marginVertical: 6,
  },
  button: {
    backgroundColor: "#3b82f6",
    paddingVertical: 14,
    borderRadius: 12,
    marginTop: 20,
    width: "90%",
    alignItems: "center",
  },
  buttonText: { color: "white", fontWeight: "bold", fontSize: 16 },
  loading: { color: "#9ca3af", marginTop: 20 },
  result: { color: "#e5e7eb", marginTop: 20, textAlign: "left" },
});
