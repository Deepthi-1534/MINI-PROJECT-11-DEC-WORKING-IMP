import { CheckCircle, AlertTriangle, Info, TrendingUp } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface CamouflageRegion {
  description: string;
  intensity: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface AnalysisData {
  detected: boolean;
  species?: string;
  camouflagePercentage: number;
  confidence: number;
  description: string;
  adaptations: string[];
  boundingBox?: BoundingBox | null;
  camouflageRegions?: CamouflageRegion[];
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

      {/* Metrics Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Camouflage Percentage */}
        <Card className="p-6 bg-card border-border/50 hover:border-accent/50 transition-all duration-300">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-semibold text-foreground">Camouflage Level</h3>
            <TrendingUp className="w-6 h-6 text-accent" />
          </div>
          <div className="space-y-3">
            <div className="flex items-end justify-between">
              <span className="text-5xl font-bold text-accent">
                {data.camouflagePercentage}%
              </span>
              <span className="text-sm text-muted-foreground mb-2">
                {data.camouflagePercentage >= 70 ? 'Excellent' : 
                 data.camouflagePercentage >= 40 ? 'Moderate' : 'Low'}
              </span>
            </div>
            <Progress 
              value={data.camouflagePercentage} 
              className="h-3"
            />
          </div>
        </Card>

        {/* Detection Confidence */}
        <Card className="p-6 bg-card border-border/50 hover:border-secondary/50 transition-all duration-300">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-semibold text-foreground">AI Confidence</h3>
            <Info className="w-6 h-6 text-secondary" />
          </div>
          <div className="space-y-3">
            <div className="flex items-end justify-between">
              <span className="text-5xl font-bold text-secondary">
                {data.confidence}%
              </span>
              <span className="text-sm text-muted-foreground mb-2">
                {data.confidence >= 80 ? 'Very High' : 
                 data.confidence >= 60 ? 'High' : 'Moderate'}
              </span>
            </div>
            <Progress 
              value={data.confidence} 
              className="h-3"
            />
          </div>
        </Card>
      </div>

      {/* Visual Analysis */}
      <Card className="p-8 bg-card border-border/50 shadow-elevated">
        <h3 className="text-2xl font-bold mb-6 text-foreground">Visual Analysis</h3>
        <div className="grid md:grid-cols-2 gap-6">
          {/* Original Image */}
          <div className="space-y-3">
            <h4 className="text-lg font-semibold text-muted-foreground">Original Image</h4>
            <div className="rounded-xl overflow-hidden border border-border/50">
              <img src={imagePreview} alt="Original" className="w-full h-auto" />
            </div>
          </div>

          {/* Heat Map with Regions */}
          <div className="space-y-3">
            <h4 className="text-lg font-semibold text-muted-foreground">Camouflage Detection Map</h4>
            <div className="relative rounded-xl overflow-hidden border border-border/50">
              <img src={imagePreview} alt="Detection map" className="w-full h-auto" />
              
              {/* Main bounding box */}
              {data.boundingBox && (
                <div
                  className="absolute border-4 border-secondary rounded-lg"
                  style={{
                    left: `${data.boundingBox.x}%`,
                    top: `${data.boundingBox.y}%`,
                    width: `${data.boundingBox.width}%`,
                    height: `${data.boundingBox.height}%`,
                  }}
                >
                  <div className="absolute -top-8 left-0 px-3 py-1 rounded-full bg-secondary text-secondary-foreground text-xs font-bold shadow-glow-accent">
                    Species Detected
                  </div>
                </div>
              )}
              
              {/* Camouflage intensity regions */}
              {data.camouflageRegions && data.camouflageRegions.map((region, idx) => (
                <div
                  key={idx}
                  className="absolute rounded-lg transition-opacity hover:opacity-90"
                  style={{
                    left: `${region.x}%`,
                    top: `${region.y}%`,
                    width: `${region.width}%`,
                    height: `${region.height}%`,
                    background: `linear-gradient(135deg, 
                      hsla(${35 + (region.intensity * 0.5)}, 90%, 60%, ${region.intensity / 200}),
                      hsla(${0 + (region.intensity * 0.3)}, 85%, 60%, ${region.intensity / 150})
                    )`,
                    border: `2px solid hsla(${35}, 90%, 60%, ${region.intensity / 100})`,
                    boxShadow: `0 0 20px hsla(${35}, 90%, 60%, ${region.intensity / 200})`,
                  }}
                  title={`${region.description}: ${region.intensity}% camouflage intensity`}
                >
                  <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 px-2 py-1 rounded bg-accent/90 text-accent-foreground text-xs font-semibold whitespace-nowrap">
                    {region.intensity}%
                  </div>
                </div>
              ))}
              
              <div className="absolute top-4 right-4 px-3 py-1 rounded-full bg-background/80 backdrop-blur-sm text-sm font-semibold">
                {data.camouflagePercentage}% Overall
              </div>
            </div>
            
            {/* Region Legend */}
            {data.camouflageRegions && data.camouflageRegions.length > 0 && (
              <div className="mt-4 p-4 rounded-lg bg-muted/50 border border-border/50">
                <h5 className="text-sm font-semibold mb-2 text-foreground">Detected Regions:</h5>
                <div className="space-y-1">
                  {data.camouflageRegions.map((region, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-sm text-muted-foreground">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{
                          background: `hsla(${35 + (region.intensity * 0.5)}, 90%, 60%, 1)`,
                          boxShadow: '0 0 8px currentColor'
                        }}
                      />
                      <span>{region.description} - {region.intensity}% intensity</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Adaptations */}
      {data.adaptations && data.adaptations.length > 0 && (
        <Card className="p-8 bg-card border-border/50 shadow-elevated">
          <h3 className="text-2xl font-bold mb-6 text-foreground">Camouflage Adaptations</h3>
          <div className="grid md:grid-cols-2 gap-4">
            {data.adaptations.map((adaptation, idx) => (
              <div
                key={idx}
                className="flex items-start gap-3 p-4 rounded-lg bg-primary/10 border border-primary/20"
              >
                <CheckCircle className="w-5 h-5 text-primary-glow flex-shrink-0 mt-0.5" />
                <p className="text-muted-foreground">{adaptation}</p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default AnalysisResults;