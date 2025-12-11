import { CheckCircle, AlertTriangle, Info, TrendingUp } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

interface BoundingBox {
  x: number;      // percentage x1
  y: number;      // percentage y1
  width: number;  // percentage width
  height: number; // percentage height
}

interface AnalysisData {
  detected: boolean;
  species?: string;
  camouflagePercentage: number;
  confidence: number;
  description: string;
  adaptations: string[];
  boundingBox?: BoundingBox | null;
}

interface AnalysisResultsProps {
  data: AnalysisData;
  imagePreview: string;
}

const AnalysisResults = ({ data, imagePreview }: AnalysisResultsProps) => {
  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">

      {/* Detection Status */}
      <Card className="p-8 bg-card border-border/50 shadow-elevated">
        <div className="flex items-center gap-4 mb-6">
          {data.detected ? (
            <CheckCircle className="w-12 h-12 text-primary-glow" />
          ) : (
            <AlertTriangle className="w-12 h-12 text-accent" />
          )}
          <div>
            <h2 className="text-3xl font-bold text-foreground">
              {data.detected ? 'Camouflage Detected!' : 'No Camouflage Detected'}
            </h2>
            {data.species && (
              <p className="text-lg text-muted-foreground mt-1">
                Species: <span className="text-secondary font-semibold">{data.species}</span>
              </p>
            )}
          </div>
        </div>

        <p className="text-muted-foreground text-lg leading-relaxed">
          {data.description}
        </p>
      </Card>

      {/* Metrics */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card className="p-6 bg-card border-border/50">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-semibold text-foreground">Camouflage Level</h3>
            <TrendingUp className="w-6 h-6 text-accent" />
          </div>
          <div className="space-y-3">
            <span className="text-5xl font-bold text-accent">
              {data.camouflagePercentage}%
            </span>
            <Progress value={data.camouflagePercentage} className="h-3" />
          </div>
        </Card>

        <Card className="p-6 bg-card border-border/50">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-semibold text-foreground">AI Confidence</h3>
            <Info className="w-6 h-6 text-secondary" />
          </div>
          <div className="space-y-3">
            <span className="text-5xl font-bold text-secondary">{data.confidence}%</span>
            <Progress value={data.confidence} className="h-3" />
          </div>
        </Card>
      </div>

      {/* Visual Analysis */}
      <Card className="p-8 bg-card border-border/50 shadow-elevated">
        <h3 className="text-2xl font-bold mb-6 text-foreground">Visual Analysis</h3>
        
        <div className="grid md:grid-cols-2 gap-6">

          {/* Original */}
          <div>
            <h4 className="text-lg font-semibold text-muted-foreground">Original Image</h4>
            <div className="rounded-xl overflow-hidden border border-border/50 mt-3">
              <img src={imagePreview} className="w-full" />
            </div>
          </div>

          {/* Detection Map */}
          <div>
            <h4 className="text-lg font-semibold text-muted-foreground">Camouflage Detection Map</h4>

            <div className="relative rounded-xl overflow-hidden border border-border/50 mt-3">
              <img id="detect-image" src={imagePreview} className="w-full h-auto" />

              {/* Draw ONLY main bounding box */}
              {data.boundingBox && (
                <div
                  className="absolute border-4 border-secondary rounded-md"
                  style={{
                    left: `${data.boundingBox.x}%`,
                    top: `${data.boundingBox.y}%`,
                    width: `${data.boundingBox.width}%`,
                    height: `${data.boundingBox.height}%`,
                  }}
                />
              )}

              <div className="absolute top-4 right-4 px-3 py-1 rounded-full bg-background/80 backdrop-blur-sm text-sm font-semibold">
                {data.camouflagePercentage}% Overall
              </div>
            </div>

            {/* Removed region legend + region overlays */}
          </div>

        </div>
      </Card>

      {/* Adaptations */}
      {data.adaptations?.length > 0 && (
        <Card className="p-8 bg-card border-border/50 shadow-elevated">
          <h3 className="text-2xl font-bold mb-6 text-foreground">Camouflage Adaptations</h3>
          <div className="grid md:grid-cols-2 gap-4">
            {data.adaptations.map((adapt, idx) => (
              <div key={idx} className="flex items-start gap-3 p-4 rounded-lg bg-primary/10">
                <CheckCircle className="w-5 h-5 text-primary-glow" />
                <p className="text-muted-foreground">{adapt}</p>
              </div>
            ))}
          </div>
        </Card>
      )}

    </div>
  );
};

export default AnalysisResults;
