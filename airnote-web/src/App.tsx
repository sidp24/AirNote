import { useEffect } from "react";
import { anonLogin } from "./firebase";
import Control from "./pages/Control";
import Dashboard from "./pages/Dashboard";

export default function App() {
  // Automatically log in anonymously to Firebase when the app starts
  useEffect(() => {
    anonLogin();
  }, []);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "360px 1fr",
        gap: 24,
        height: "100vh",
        overflow: "hidden",
        fontFamily: "Inter, system-ui, sans-serif",
      }}
    >
      {/* Left side: control panel */}
      <div
        style={{
          borderRight: "1px solid #eee",
          overflowY: "auto",
          padding: "16px",
          backgroundColor: "#fafafa",
        }}
      >
        <Control />
      </div>

      {/* Right side: collaborative dashboard */}
      <div style={{ overflowY: "auto", padding: "16px" }}>
        <Dashboard teamId="demo" folderId="main" />
      </div>
    </div>
  );
}