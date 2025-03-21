
import React, { useEffect, useRef } from "react";
import { Chart, BarController, CategoryScale, LinearScale, BarElement, Tooltip, Legend } from "chart.js";

Chart.register(BarController, CategoryScale, LinearScale, BarElement, Tooltip, Legend);

type ModelData = {
  name: string;
  accuracy: number;
  precision: number;
};

interface ModelPerformanceChartProps {
  models: ModelData[];
  className?: string;
}

const ModelPerformanceChart: React.FC<ModelPerformanceChartProps> = ({ models, className = "" }) => {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);

  useEffect(() => {
    if (chartRef.current) {
      // Destroy previous chart instance if it exists
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }

      const ctx = chartRef.current.getContext("2d");
      if (ctx) {
        // Extract data for the chart
        const labels = models.map((model) => model.name);
        const accuracyData = models.map((model) => model.accuracy * 100);
        const precisionData = models.map((model) => model.precision * 100);

        // Create the chart
        chartInstance.current = new Chart(ctx, {
          type: "bar",
          data: {
            labels: labels,
            datasets: [
              {
                label: "Accuracy",
                data: accuracyData,
                backgroundColor: "rgba(99, 102, 241, 0.7)",
                borderColor: "rgba(99, 102, 241, 1)",
                borderWidth: 1,
                borderRadius: 4,
              },
              {
                label: "Precision",
                data: precisionData,
                backgroundColor: "rgba(52, 211, 153, 0.7)",
                borderColor: "rgba(52, 211, 153, 1)",
                borderWidth: 1,
                borderRadius: 4,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: {
                position: "bottom",
                labels: {
                  font: {
                    family: "'Inter', sans-serif",
                    size: 12,
                  },
                  color: "#4b5563",
                  usePointStyle: true,
                  padding: 20,
                },
              },
              tooltip: {
                backgroundColor: "rgba(255, 255, 255, 0.9)",
                titleColor: "#111827",
                bodyColor: "#374151",
                borderColor: "rgba(229, 231, 235, 1)",
                borderWidth: 1,
                padding: 10,
                boxPadding: 4,
                usePointStyle: true,
                callbacks: {
                  label: function (context) {
                    return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
                  },
                },
                bodyFont: {
                  family: "'Inter', sans-serif",
                },
                titleFont: {
                  family: "'Inter', sans-serif",
                  weight: "600",
                },
              },
            },
            scales: {
              x: {
                grid: {
                  display: false,
                },
                ticks: {
                  font: {
                    family: "'Inter', sans-serif",
                  },
                  color: "#6b7280",
                },
              },
              y: {
                beginAtZero: true,
                max: 100,
                grid: {
                  color: "rgba(243, 244, 246, 1)",
                },
                border: {
                  dash: [4, 4],
                },
                ticks: {
                  font: {
                    family: "'Inter', sans-serif",
                  },
                  color: "#6b7280",
                  callback: function (value) {
                    return value + "%";
                  },
                },
              },
            },
          },
        });
      }
    }

    // Cleanup function
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [models]);

  return (
    <div className={`w-full h-full ${className}`}>
      <canvas ref={chartRef} />
    </div>
  );
};

export default ModelPerformanceChart;
