
import React from "react";

const ChartIcon: React.FC<{ className?: string }> = ({ className = "w-5 h-5" }) => {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M8 16H4C3.44772 16 3 15.5523 3 15V9C3 8.44772 3.44772 8 4 8H8C8.55228 8 9 8.44772 9 9V15C9 15.5523 8.55228 16 8 16Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M14 19H10C9.44772 19 9 18.5523 9 18V6C9 5.44772 9.44772 5 10 5H14C14.5523 5 15 5.44772 15 6V18C15 18.5523 14.5523 19 14 19Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M20 16H16C15.4477 16 15 15.5523 15 15V9C15 8.44772 15.4477 8 16 8H20C20.5523 8 21 8.44772 21 9V15C21 15.5523 20.5523 16 20 16Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

export default ChartIcon;
