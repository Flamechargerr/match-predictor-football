
import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import FootballIcon from "@/components/FootballIcon";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-background to-background/80 px-4">
      <div className="text-center animate-fade-in">
        <FootballIcon className="w-16 h-16 text-primary mx-auto mb-6" />
        <h1 className="text-6xl font-bold text-gray-900 mb-4">404</h1>
        <p className="text-xl text-muted-foreground mb-8">
          Oops! This page went out of bounds
        </p>
        <Button 
          asChild 
          size="lg" 
          className="bg-gradient-to-r from-home-DEFAULT to-away-DEFAULT hover:from-home-dark hover:to-away-dark transition-all duration-300"
        >
          <a href="/">Return to Home Field</a>
        </Button>
      </div>
    </div>
  );
};

export default NotFound;
