"use client";

import { useState, useRef, useEffect, ReactNode } from "react";

interface ResizablePanelProps {
  leftPanel: ReactNode;
  rightPanel: ReactNode;
  defaultRightWidth?: number;
  minRightWidth?: number;
  maxRightWidth?: number;
  showRightPanel: boolean;
}

export default function ResizablePanel({
  leftPanel,
  rightPanel,
  defaultRightWidth = 600,
  minRightWidth = 400,
  maxRightWidth = 900,
  showRightPanel,
}: ResizablePanelProps) {
  const [rightWidth, setRightWidth] = useState(defaultRightWidth);
  const [isResizing, setIsResizing] = useState(false);
  const [isLargeScreen, setIsLargeScreen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Handle responsive behavior
  useEffect(() => {
    const checkScreenSize = () => {
      setIsLargeScreen(window.innerWidth >= 1024);
    };

    checkScreenSize();
    window.addEventListener("resize", checkScreenSize);

    return () => {
      window.removeEventListener("resize", checkScreenSize);
    };
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing || !containerRef.current) return;

      const containerRect = containerRef.current.getBoundingClientRect();
      const newWidth = containerRect.right - e.clientX;

      // Clamp the width between min and max
      const clampedWidth = Math.min(
        Math.max(newWidth, minRightWidth),
        maxRightWidth
      );

      setRightWidth(clampedWidth);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    if (isResizing) {
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, minRightWidth, maxRightWidth]);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  return (
    <div
      ref={containerRef}
      className="h-full flex flex-col lg:flex-row relative"
    >
      {/* Left Panel — always mounted so chat state is preserved */}
      <div className="flex-1 overflow-hidden border-b lg:border-b-0 border-gray-200">
        {leftPanel}
      </div>

      {/* Resize Handle — hidden when right panel is closed */}
      {showRightPanel && (
        <div
          onMouseDown={handleMouseDown}
          className="hidden lg:block w-1 bg-gray-200 hover:bg-blue-500 cursor-col-resize transition-colors relative group"
          style={{ flexShrink: 0 }}
        >
          <div className="absolute inset-y-0 -left-1 -right-1 group-hover:bg-blue-500/20 transition-colors" />
        </div>
      )}

      {/* Right Panel — hidden via CSS, not unmounted */}
      <div
        className="bg-white border-t lg:border-t-0 lg:border-l border-gray-200 h-80 lg:h-full shrink-0 overflow-hidden"
        style={{
          width: showRightPanel ? (isLargeScreen ? `${rightWidth}px` : "100%") : "0px",
          display: showRightPanel ? undefined : "none",
        }}
      >
        {rightPanel}
      </div>
    </div>
  );
}

