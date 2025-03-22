
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useState, useEffect } from "react";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

const App = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingMessage, setLoadingMessage] = useState("Initializing Python environment...");

  useEffect(() => {
    // Simulate loading progress
    const progressInterval = setInterval(() => {
      setLoadingProgress(prev => {
        if (prev >= 95) {
          clearInterval(progressInterval);
          return prev;
        }
        const increment = Math.random() * 10;
        const newProgress = Math.min(95, prev + increment);
        
        // Update loading messages based on progress
        if (newProgress > 80 && prev <= 80) {
          setLoadingMessage("Finalizing initialization...");
        } else if (newProgress > 50 && prev <= 50) {
          setLoadingMessage("Loading machine learning libraries...");
        } else if (newProgress > 20 && prev <= 20) {
          setLoadingMessage("Setting up Python environment...");
        }
        
        return newProgress;
      });
    }, 700);

    // Add script tag for Pyodide
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/pyodide/v0.23.4/full/pyodide.js';
    script.async = true;
    script.onload = () => {
      // Just set loading to false, the actual initialization will happen in the service
      setLoadingProgress(100);
      setLoadingMessage("Ready! Loading application...");
      setTimeout(() => {
        setIsLoading(false);
        clearInterval(progressInterval);
      }, 500);
    };
    document.head.appendChild(script);

    return () => {
      // Clean up script tag on unmount
      clearInterval(progressInterval);
      if (document.head.contains(script)) {
        document.head.removeChild(script);
      }
    };
  }, []);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-indigo-50 to-purple-50">
        <div className="text-2xl font-bold text-indigo-700 mb-4">Football Match Predictor</div>
        <div className="text-lg text-gray-600 mb-6">{loadingMessage}</div>
        <div className="w-64 h-3 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 transition-all duration-500 ease-out"
            style={{ width: `${loadingProgress}%` }}
          ></div>
        </div>
        <div className="mt-4 text-sm text-gray-500">
          {loadingProgress < 100 ? 
            "This might take a few seconds on first load..." :
            "Almost there! Starting the application..."}
        </div>
      </div>
    );
  }

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
