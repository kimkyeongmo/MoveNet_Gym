import React from "react";
import { View, TouchableOpacity, Text, StyleSheet } from "react-native";
//import Ionicons from "@expo/vector-icons/Ionicons";

interface Props {
  current: string;
  onChange: (screen: string) => void;
}

export default function BottomBar({ current, onChange }: Props) {
  return (
    <View style={styles.bar}>
      {tabs.map((tab) => (
        <TouchableOpacity
          key={tab.key}
          style={styles.tab}
          onPress={() => onChange(tab.key)}
        >
          {/* <Ionicons
            name={tab.icon}
            size={22}
            color={current === tab.key ? "#3b82f6" : "#ccc"}
          /> */}
          <Text
            style={[
              styles.label,
              { color: current === tab.key ? "#3b82f6" : "#ccc" },
            ]}
          >
            {tab.label}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

// 🔹 탭 구성 (3개만 남김)
const tabs = [
  { key: "home", label: "홈", icon: "home" },
  { key: "coach", label: "AI 코치", icon: "fitness" },
  { key: "plan", label: "식단+루틴", icon: "restaurant" },
];

const styles = StyleSheet.create({
  bar: {
    flexDirection: "row",
    justifyContent: "space-around",
    backgroundColor: "#0d1117",
    borderTopWidth: 1,
    borderTopColor: "#222",
    paddingVertical: 10,
  },
  tab: { alignItems: "center" },
  label: { fontSize: 12, marginTop: 2 },
});
