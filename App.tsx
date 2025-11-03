import React, { useState } from "react";
import { View, StyleSheet } from "react-native";
import HomeScreen from "./app/src/screens/HomeScreen";
import CoachScreen from "./app/src/screens/CoachScreen";
import PlanScreen from "./app/src/screens/PlanScreen";
import BottomBar from "./app/src/components/BottomBar";

export default function App() {
  const [currentScreen, setCurrentScreen] = useState("home");

  const renderScreen = () => {
    switch (currentScreen) {
      case "home":
        return <HomeScreen navigate={setCurrentScreen} />;
      case "coach":
        return <CoachScreen navigate={setCurrentScreen} />;
      case "plan":
        return <PlanScreen navigate={setCurrentScreen} />;
      default:
        return <HomeScreen navigate={setCurrentScreen} />;
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.content}>{renderScreen()}</View>
      <BottomBar current={currentScreen} onChange={setCurrentScreen} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0d1117" },
  content: { flex: 1 },
});
