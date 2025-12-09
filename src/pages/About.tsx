import { Microscope, Brain, Target, Shield } from 'lucide-react';
import Navigation from '@/components/Navigation';

const About = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="pt-32 pb-20">
        <div className="container mx-auto px-6">
          {/* Header */}
          <div className="max-w-4xl mx-auto text-center mb-16 space-y-6">
            <h1 className="text-5xl md:text-6xl font-bold">
              About
              <span className="block bg-gradient-to-r from-primary-glow to-secondary bg-clip-text text-transparent">
                CamoDetect 
              </span>
            </h1>
            <p className="text-xl text-muted-foreground leading-relaxed">
              Pioneering the intersection of artificial intelligence and biological research 
              to understand nature's most sophisticated survival mechanisms.
            </p>
          </div>

          {/* Mission Section */}
          <div className="max-w-5xl mx-auto mb-20">
            <div className="p-10 rounded-3xl bg-card border border-border/50 shadow-elevated">
              <h2 className="text-3xl font-bold mb-6 flex items-center gap-3">
                <Target className="w-8 h-8 text-primary-glow" />
                Our Mission
              </h2>
              <p className="text-lg text-muted-foreground leading-relaxed mb-4">
                Camouflage represents millions of years of evolutionary refinement—a testament to nature's 
                problem-solving capabilities. CamoDetect AI exists to decode these biological masterpieces, 
                making invisible adaptation visible through cutting-edge machine learning.
              </p>
              <p className="text-lg text-muted-foreground leading-relaxed">
                We believe that understanding how animals achieve perfect concealment can inspire breakthrough 
                innovations in materials science, computer vision, and adaptive systems across multiple industries.
              </p>
            </div>
          </div>

          {/* How It Works */}
          <div className="mb-20">
            <h2 className="text-4xl font-bold text-center mb-12">How It Works</h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                {
                  icon: Microscope,
                  title: 'Image Analysis',
                  description: 'Advanced vision models scan every pixel, analyzing color, texture, and pattern distributions',
                },
                {
                  icon: Brain,
                  title: 'Pattern Recognition',
                  description: 'Neural networks trained on thousands of species identify biological features and adaptive strategies',
                },
                {
                  icon: Shield,
                  title: 'Camouflage Detection',
                  description: 'Proprietary algorithms isolate concealed organisms from complex environmental backgrounds',
                },
                {
                  icon: Target,
                  title: 'Visualization',
                  description: 'Heat maps and overlays reveal detection certainty and highlight camouflaged regions',
                },
              ].map((step, idx) => (
                <div
                  key={idx}
                  className="p-6 rounded-2xl bg-card border border-border/50 hover:border-primary/50 transition-all duration-300 hover:shadow-elevated group"
                >
                  <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center mb-4 group-hover:shadow-glow-primary transition-all duration-300">
                    <step.icon className="w-6 h-6 text-primary-glow" />
                  </div>
                  <h3 className="text-xl font-bold mb-3">{step.title}</h3>
                  <p className="text-muted-foreground">{step.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Technology Stack */}
          <div className="max-w-5xl mx-auto">
            <div className="p-10 rounded-3xl bg-gradient-hero/20 border border-primary/30 shadow-elevated">
              <h2 className="text-3xl font-bold mb-6">Technology Behind the Magic</h2>
              <div className="grid md:grid-cols-2 gap-8 text-muted-foreground">
                <div>
                  <h3 className="text-xl font-semibold text-foreground mb-3">AI Models</h3>
                  <ul className="space-y-2 list-disc list-inside">
                    <li>Advanced object detection and segmentation</li>
                    <li>Transfer learning from biodiversity datasets</li>
                    <li>Real-time inference with sub-second latency</li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-foreground mb-3">Visualization</h3>
                  <ul className="space-y-2 list-disc list-inside">
                    
                    <li>Multi-layer overlay composition</li>
                    <li>Confidence-weighted color mapping</li>
                    <li>Interactive region highlighting</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Applications */}
          <div className="mt-20 text-center">
            <h2 className="text-4xl font-bold mb-8">Real-World Applications</h2>
            <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
              {[
                'Wildlife Conservation & Monitoring',
                'Biomimetic Design & Engineering',
                'Computer Vision Research',
              ].map((app, idx) => (
                <div
                  key={idx}
                  className="p-6 rounded-2xl bg-card border border-border/50 hover:border-secondary/50 transition-all duration-300 hover:shadow-glow-accent"
                >
                  <p className="text-lg font-semibold text-foreground">{app}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;