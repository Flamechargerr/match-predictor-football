
import React from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";

type ModelData = {
  name: string;
  accuracy: number;
  precision: number;
  f1Score?: number;
};

interface ModelPerformanceChartProps {
  models: ModelData[];
  className?: string;
}

const ModelPerformanceChart: React.FC<ModelPerformanceChartProps> = ({ models, className = "" }) => {
  // Prepare data for the chart
  const chartData = models.map((model) => ({
    name: model.name,
    Accuracy: Number((model.accuracy * 100).toFixed(1)),
    Precision: Number((model.precision * 100).toFixed(1)),
    F1Score: model.f1Score ? Number((model.f1Score * 100).toFixed(1)) : undefined,
  }));

  // Custom colors for bars with gradients
  const accuracyColors = ["rgba(99, 102, 241, 0.8)", "rgba(79, 70, 229, 0.8)"];
  const precisionColors = ["rgba(16, 185, 129, 0.8)", "rgba(5, 150, 105, 0.8)"];
  const f1ScoreColors = ["rgba(236, 72, 153, 0.8)", "rgba(219, 39, 119, 0.8)"];

  return (
    <div className={`w-full h-full ${className}`}>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={chartData}
          margin={{
            top: 20,
            right: 20,
            left: 20,
            bottom: 15,
          }}
          barSize={24}
          barGap={8}
        >
          <defs>
            <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#4F46E5" />
              <stop offset="100%" stopColor="#818CF8" />
            </linearGradient>
            <linearGradient id="precisionGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10B981" />
              <stop offset="100%" stopColor="#34D399" />
            </linearGradient>
            <linearGradient id="f1ScoreGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#EC4899" />
              <stop offset="100%" stopColor="#F472B6" />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
          <XAxis 
            dataKey="name" 
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12, fill: "#6b7280", fontFamily: "'Inter', sans-serif" }}
            dy={10}
          />
          <YAxis 
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12, fill: "#6b7280", fontFamily: "'Inter', sans-serif" }}
            tickFormatter={(value) => `${value}%`}
            domain={[0, 100]}
          />
          <Tooltip
            contentStyle={{ 
              backgroundColor: "rgba(255, 255, 255, 0.95)",
              border: "1px solid rgba(229, 231, 235, 1)",
              borderRadius: "0.5rem",
              boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
              fontSize: "12px",
              fontFamily: "'Inter', sans-serif",
              padding: "8px 12px",
              color: "#374151",
            }}
            formatter={(value: number) => [`${value.toFixed(1)}%`, undefined]}
            labelStyle={{ fontWeight: 600, color: "#111827", marginBottom: "4px" }}
            cursor={{ fill: 'rgba(0,0,0,0.05)' }}
          />
          <Legend 
            wrapperStyle={{ paddingTop: "12px", fontSize: "12px", fontFamily: "'Inter', sans-serif" }}
            iconType="circle"
            align="center"
          />
          <Bar 
            dataKey="Accuracy" 
            fill="url(#accuracyGradient)" 
            radius={[4, 4, 0, 0]} 
            animationDuration={1500}
            name="Accuracy"
          />
          <Bar 
            dataKey="Precision" 
            fill="url(#precisionGradient)" 
            radius={[4, 4, 0, 0]} 
            animationDuration={1500}
            animationBegin={300}
            name="Precision"
          />
          {chartData[0].F1Score !== undefined && (
            <Bar 
              dataKey="F1Score" 
              fill="url(#f1ScoreGradient)" 
              radius={[4, 4, 0, 0]} 
              animationDuration={1500}
              animationBegin={600}
              name="F1 Score"
            />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ModelPerformanceChart;
