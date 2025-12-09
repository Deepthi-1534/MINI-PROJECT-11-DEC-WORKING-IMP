import { useCallback, useState, useRef } from 'react';
import { Upload, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

interface ImageUploadProps {
  onImageSelect: (file: File, preview: string) => void;
  isAnalyzing: boolean;
}

const ImageUpload = ({ onImageSelect, isAnalyzing }: ImageUploadProps) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      toast({
        title: 'Invalid file type',
        description: 'Please upload an image file',
        variant: 'destructive',
      });
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setPreview(result);
      onImageSelect(file, result);
    };
    reader.readAsDataURL(file);
  }, [onImageSelect, toast]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const clearImage = useCallback(() => {
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  const triggerFileInput = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <div className="space-y-6">
      <div
        className={`relative border-2 border-dashed rounded-2xl p-12 transition-all duration-300 ${
          isDragging
            ? 'border-primary-glow bg-primary/10 shadow-glow-primary'
            : preview
            ? 'border-primary/50 bg-card'
            : 'border-border hover:border-primary/50 bg-card/50'
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        {preview ? (
          <div className="relative">
            <img
              src={preview}
              alt="Upload preview"
              className="w-full h-auto rounded-xl shadow-elevated"
            />
            {!isAnalyzing && (
              <Button
                variant="destructive"
                size="icon"
                className="absolute top-4 right-4 shadow-lg"
                onClick={clearImage}
              >
                <X className="w-4 h-4" />
              </Button>
            )}
          </div>
        ) : (
          <div className="text-center space-y-4">
            <div className="w-20 h-20 mx-auto rounded-full bg-primary/20 flex items-center justify-center">
              <Upload className="w-10 h-10 text-primary-glow" />
            </div>
            <div className="space-y-2">
              <p className="text-xl font-semibold text-foreground">
                Upload an Image
              </p>
              <p className="text-muted-foreground">
                Drag and drop or click to select
              </p>
            </div>
            <Button
              onClick={triggerFileInput}
              className="bg-gradient-hero hover:shadow-glow-primary transition-all duration-300"
              disabled={isAnalyzing}
            >
              Select File
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileInput}
              disabled={isAnalyzing}
            />
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-muted-foreground">
        <div className="p-3 rounded-lg bg-card border border-border/50 text-center">
          <p className="font-semibold text-foreground">Supported</p>
          <p>JPG, PNG, WEBP</p>
        </div>
        <div className="p-3 rounded-lg bg-card border border-border/50 text-center">
          <p className="font-semibold text-foreground">Max Size</p>
          <p>10MB</p>
        </div>
        <div className="p-3 rounded-lg bg-card border border-border/50 text-center">
          <p className="font-semibold text-foreground">Processing</p>
          <p>~2-5 seconds</p>
        </div>
        <div className="p-3 rounded-lg bg-card border border-border/50 text-center">
          <p className="font-semibold text-foreground">AI Model</p>
          <p>Gemini 2.5 Pro</p>
        </div>
      </div>
    </div>
  );
};

export default ImageUpload;