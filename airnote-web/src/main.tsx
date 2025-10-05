import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider, Outlet, Link } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Alive from "./pages/Alive";
import Note from "./pages/Note";
import "./index.css";
import { Sparkles } from "lucide-react";

function Shell() {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 bg-[rgba(11,16,32,.6)] backdrop-blur-[2px] border-b border-[var(--border)]">
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center gap-3">
          <Link to="/" className="btn">
            <Sparkles className="size-4" />
            <span className="font-semibold">AirNote</span>
          </Link>
          <nav className="ml-auto flex items-center gap-2">
            <Link to="/" className="btn">Notes</Link>
            <Link to="/alive" className="btn-accent">Alive</Link>
          </nav>
        </div>
      </header>
      <main className="px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}

const router = createBrowserRouter([
  { path: "/", element: <Shell />, children: [
      { index: true, element: <Dashboard /> },
      { path: "alive", element: <Alive /> },
      { path: "note/:id", element: <Note /> },
  ]},
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);