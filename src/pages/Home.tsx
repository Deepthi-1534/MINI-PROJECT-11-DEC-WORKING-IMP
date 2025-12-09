import { Link } from 'react-router-dom';
import { ArrowRight, Scan, Zap, Target } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Navigation from '@/components/Navigation';
import heroImage from '@/assets/hero-chameleon.jpg';

const Home = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-hero opacity-30" />
        <div className="absolute inset-0" style={{
          backgroundImage: `url(${heroImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          opacity: 0.15,
        }} />
        
        <div className="container mx-auto px-6 relative z-10">
          <div className="max-w-4xl mx-auto text-center space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-1000">
            <h1 className="text-6xl md:text-7xl font-bold leading-tight">
              Image Analysis 
              <span className="block bg-gradient-to-r from-primary-glow via-secondary to-accent bg-clip-text text-transparent animate-pulse">
                using Camouflage detection
              </span>
            </h1>
            
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Advanced AI-powered camouflage detection system. Upload any image and watch as our neural networks 
              reveal the invisible, analyze adaptive coloration, and quantify nature's most sophisticated survival strategies.
            </p>

            <div className="flex items-center justify-center gap-4 pt-4">
              <Link to="/detect">
                <Button 
                  size="lg" 
                  className="bg-gradient-hero hover:shadow-glow-primary transition-all duration-300 hover:scale-105 text-lg px-8 py-6"
                >
                  Start Detecting
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
              </Link>
              <Link to="/about">
                <Button 
                  size="lg" 
                  variant="outline" 
                  className="border-primary/50 hover:border-primary hover:bg-primary/10 text-lg px-8 py-6 transition-all duration-300"
                >
                  Learn More
                </Button>
              </Link>
            </div>
          </div>
        </div>

        {/* Animated background elements */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-primary-glow/20 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-secondary/20 rounded-full blur-3xl animate-pulse delay-1000" />
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 border-t border-border/50">
        <div className="container mx-auto px-6">
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: Scan,
                title: 'AI-Powered Analysis',
                description: 'State-of-the-art vision models detect even the most sophisticated camouflage patterns',
              },
              {
                icon: Target,
                title: ' Visualization',
                description: 'See exactly where camouflage occurs with advanced overlay and highlighting technology',
              },
              {
                icon: Zap,
                title: 'Instant Results',
                description: 'Get detailed analysis including species identification and camouflage percentage in seconds',
              },
            ].map((feature, idx) => (
              <div
                key={idx}
                className="group p-8 rounded-2xl bg-card border border-border/50 hover:border-primary/50 transition-all duration-300 hover:shadow-elevated hover:-translate-y-1"
                style={{ animationDelay: `${idx * 100}ms` }}
              >
                <div className="w-14 h-14 rounded-xl bg-primary/20 flex items-center justify-center mb-6 group-hover:shadow-glow-primary transition-all duration-300 group-hover:scale-110">
                  <feature.icon className="w-7 h-7 text-primary-glow" />
                </div>
                <h3 className="text-2xl font-bold mb-3 text-foreground">{feature.title}</h3>
                <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 border-t border-border/50">
        <div className="container mx-auto px-6">
          <div className="max-w-3xl mx-auto text-center space-y-6 p-12 rounded-3xl bg-gradient-hero/20 border border-primary/30 shadow-elevated">
            <h2 className="text-4xl font-bold text-foreground">
              Ready to Reveal the Invisible?
            </h2>
            <p className="text-lg text-muted-foreground">
              Upload your first image and experience the power of advanced camouflage detection
            </p>
            <Link to="/detect">
              <Button 
                size="lg" 
                className="bg-primary hover:bg-primary-glow text-primary-foreground hover:shadow-glow-primary transition-all duration-300 hover:scale-105 mt-4"
              >
                Begin Analysis
                <ArrowRight className="w-5 h-5 ml-2" />
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;