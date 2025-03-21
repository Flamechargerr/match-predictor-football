
import React from "react";

const FootballIcon: React.FC<{ className?: string }> = ({ className = "w-8 h-8" }) => {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <circle cx="12" cy="12" r="10" fill="currentColor" fillOpacity="0.2" />
      <path
        d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 4.07C16.94 4.56 20 7.92 20 12C20 13.82 19.38 15.5 18.31 16.83L16.33 15.29C15.37 14.57 14.14 14.57 13.17 15.29L12 16.17L10.83 15.29C9.86 14.57 8.63 14.57 7.67 15.29L5.69 16.83C4.62 15.5 4 13.82 4 12C4 7.92 7.06 4.56 11 4.07V8C11 8.55 11.45 9 12 9C12.55 9 13 8.55 13 8V4.07Z"
        fill="currentColor"
      />
      <path
        d="M7.67 15.29L5.69 16.83C6.77 18.16 8.38 19.06 10.2 19.42L12 20L13.8 19.42C15.62 19.06 17.23 18.16 18.31 16.83L16.33 15.29C15.37 14.57 14.14 14.57 13.17 15.29L12 16.17L10.83 15.29C9.86 14.57 8.63 14.57 7.67 15.29Z"
        fill="currentColor"
      />
    </svg>
  );
};

export default FootballIcon;
