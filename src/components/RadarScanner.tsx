import { useEffect, useState } from 'react';
import radarPattern from '@/assets/radar-pattern.jpg';

interface RadarScannerProps {
  isScanning: boolean;
}

const RadarScanner = ({ isScanning }: RadarScannerProps) => {
  const [rotation, setRotation] = useState(0);

  useEffect(() => {
    if (!isScanning) return;

    const interval = setInterval(() => {
      setRotation((prev) => (prev + 2) % 360);
    }, 30);

    return () => clearInterval(interval);
  }, [isScanning]);

  if (!isScanning) return null;

  return (
    <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-xl flex items-center justify-center">
      <div className="relative">
        {/* Radar circles */}
        <div className="relative w-80 h-80">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="absolute inset-0 border-2 border-primary-glow/30 rounded-full animate-pulse"
              style={{
                transform: `scale(${i * 0.25})`,
                animationDelay: `${i * 0.2}s`,
              }}
            />
          ))}
          
          {/* Center radar pattern */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div
              className="w-64 h-64 rounded-full overflow-hidden opacity-40"
              style={{
                backgroundImage: `url(${radarPattern})`,
                backgroundSize: 'cover',
                backgroundPosition: 'center',
                transform: `rotate(${rotation}deg)`,
              }}
            />
          </div>

          {/* Scanning line */}
          <div
            className="absolute top-1/2 left-1/2 w-1 h-40 -mt-40 -ml-0.5 bg-gradient-to-b from-transparent via-secondary to-transparent origin-bottom"
            style={{ transform: `rotate(${rotation}deg)` }}
          />

          {/* Center dot */}
          <div className="absolute top-1/2 left-1/2 w-4 h-4 -mt-2 -ml-2 bg-secondary rounded-full shadow-glow-accent animate-pulse" />
        </div>

        {/* Status text */}
        <div className="mt-12 text-center space-y-4">
          <h3 className="text-2xl font-bold text-primary-glow">Analyzing Image</h3>
          <p className="text-muted-foreground">AI vision models processing...</p>
          <div className="flex items-center justify-center gap-2">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="w-2 h-2 bg-secondary rounded-full animate-pulse"
                style={{ animationDelay: `${i * 0.2}s` }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RadarScanner;