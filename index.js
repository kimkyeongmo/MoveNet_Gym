import { AppRegistry } from 'react-native';
import App from './App';
 import { name as appName } from './app.json';

// // 🔹 app.json이 없다면 아래처럼 직접 이름 지정해도 됩니다.
AppRegistry.registerComponent('MoveNet_Gym', () => App);
