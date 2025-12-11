import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import Navigation from '@/components/Navigation';
import ImageUpload from '@/components/ImageUpload';
import RadarScanner from '@/components/RadarScanner';
import AnalysisResults from '@/components/AnalysisResults';

const Detect = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const { toast } = useToast();

  const handleImageSelect = (file: File, preview: string) => {
    setSelectedFile(file);
    setImagePreview(preview);
    setAnalysisData(null);
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Backend error");

      const data = await response.json();
      console.log("Analysis Data:", data); // Debug

      setAnalysisData(data);

      toast({
        title: "Analysis Complete",
        description: "Camouflage detection finished successfully",
      });
    } catch (error) {
      console.error("Analysis error:", error);
      toast({
        title: "Analysis Failed",
        description: "An error occurred during analysis. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <RadarScanner isScanning={isAnalyzing} />

      <div className="pt-32 pb-20">
        <div className="container mx-auto px-6">
          <div className="max-w-6xl mx-auto space-y-12">

            <div className="text-center space-y-4">
              <h1 className="text-5xl font-bold">
                <span className="bg-gradient-to-r from-primary-glow to-secondary bg-clip-text text-transparent">
                  Camouflage Detection
                </span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                Upload an image and let our AI reveal hidden wildlife
              </p>
            </div>

            <ImageUpload 
              onImageSelect={handleImageSelect}
              isAnalyzing={isAnalyzing}
            />

            {imagePreview && !analysisData && (
              <div className="flex justify-center">
                <Button
                  size="lg"
                  onClick={analyzeImage}
                  disabled={isAnalyzing}
                  className="bg-gradient-hero hover:shadow-glow-primary transition-all duration-300 hover:scale-105 text-lg px-12 py-6"
                >
                  {isAnalyzing ? "Analyzing..." : "Analyze Image"}
                </Button>
              </div>
            )}

            {analysisData && (
              <AnalysisResults 
                data={analysisData}
                imagePreview={imagePreview}
              />
            )}

          </div>
        </div>
      </div>
    </div>
  );
};

export default Detect;
