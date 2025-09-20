import { motion } from "framer-motion";
import { Github, Linkedin, Mail, Code, Database, Palette, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const About = () => {
  const teamMembers = [
    {
      name: "Dishu Mahajan",
      role: "Full Stack Developer & AI Integration",
      description: "Led the development of AI-powered resume analysis algorithms and full-stack architecture. Specialized in machine learning integration and backend optimization.",
      skills: ["React", "Node.js", "Python", "AI/ML", "System Architecture"],
      github: "https://github.com/dishuma", // Placeholder - update with actual
      linkedin: "https://linkedin.com/in/dishuma", // Placeholder - update with actual
      email: "dishu.mahajan@example.com", // Placeholder - update with actual
      avatar: "D",
      color: "from-blue-500 to-purple-600"
    },
    {
      name: "Siddarth P Nair",
      role: "Backend Developer & Database Architect",
      description: "Designed and implemented the robust database architecture and backend APIs. Focused on performance optimization and scalable data management solutions.",
      skills: ["Node.js", "PostgreSQL", "API Design", "Cloud Architecture", "DevOps"],
      github: "https://github.com/siddarthpnair", // Placeholder - update with actual
      linkedin: "https://linkedin.com/in/siddarthpnair", // Placeholder - update with actual
      email: "siddarth.nair@example.com", // Placeholder - update with actual
      avatar: "S",
      color: "from-green-500 to-teal-600"
    },
    {
      name: "Eshwar J",
      role: "Frontend Developer & UI/UX Designer",
      description: "Crafted the premium iOS-inspired user interface with glassmorphism effects. Specialized in responsive design and smooth user experience optimization.",
      skills: ["React", "TypeScript", "Tailwind CSS", "UI/UX Design", "Framer Motion"],
      github: "https://github.com/eshwarj", // Placeholder - update with actual
      linkedin: "https://linkedin.com/in/eshwarj", // Placeholder - update with actual
      email: "eshwar.j@example.com", // Placeholder - update with actual
      avatar: "E",
      color: "from-purple-500 to-pink-600"
    }
  ];

  const features = [
    {
      icon: Code,
      title: "AI-Powered Analysis",
      description: "Advanced machine learning algorithms analyze resume content against job requirements for precise matching."
    },
    {
      icon: Database,
      title: "Scalable Architecture",
      description: "Built on modern cloud infrastructure with PostgreSQL and Supabase for reliability and performance."
    },
    {
      icon: Palette,
      title: "Premium Design",
      description: "iOS-inspired glassmorphism interface with smooth animations and responsive design across all devices."
    }
  ];

  return (
    <div className="min-h-screen pt-24 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl sm:text-5xl font-bold mb-6">
            About <span className="bg-gradient-to-r from-primary to-primary-glow bg-clip-text text-transparent">ResumeCheck</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            ResumeCheck is an innovative AI-powered platform designed to revolutionize the hiring process. 
            Our intelligent system analyzes resumes against job requirements, providing instant relevance 
            scores and actionable improvement suggestions for both candidates and recruiters.
          </p>
        </motion.div>

        {/* Mission Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mb-20"
        >
          <Card className="card-premium bg-gradient-hero text-white overflow-hidden relative">
            <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full blur-3xl -translate-y-32 translate-x-32"></div>
            <CardContent className="p-12 relative z-10">
              <div className="max-w-4xl mx-auto text-center">
                <h2 className="text-3xl sm:text-4xl font-bold mb-6">Our Mission</h2>
                <p className="text-xl opacity-90 leading-relaxed mb-8">
                  To bridge the gap between talented candidates and their dream jobs by providing 
                  intelligent, data-driven insights that benefit both job seekers and hiring teams. 
                  We believe that everyone deserves a fair chance to showcase their potential.
                </p>
                <div className="grid md:grid-cols-3 gap-8">
                  {features.map((feature, index) => (
                    <motion.div
                      key={feature.title}
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.6, delay: index * 0.1 }}
                      viewport={{ once: true }}
                      className="text-center"
                    >
                      <div className="w-16 h-16 bg-white/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                        <feature.icon className="w-8 h-8 text-white" />
                      </div>
                      <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                      <p className="opacity-80">{feature.description}</p>
                    </motion.div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Team Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mb-20"
        >
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold mb-6">Meet Our Team</h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              A passionate team of developers and designers committed to creating innovative solutions 
              that make a real difference in people's careers.
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            {teamMembers.map((member, index) => (
              <motion.div
                key={member.name}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="group"
              >
                <Card className="card-premium hover:scale-105 transition-all duration-500 overflow-hidden">
                  <CardContent className="p-8 text-center">
                    {/* Avatar */}
                    <div className="relative mb-6">
                      <div className={`w-24 h-24 bg-gradient-to-br ${member.color} rounded-full flex items-center justify-center mx-auto mb-4 text-white text-2xl font-bold shadow-lg`}>
                        {member.avatar}
                      </div>
                      <div className="absolute inset-0 w-24 h-24 mx-auto bg-gradient-to-br from-primary/20 to-primary-glow/20 rounded-full blur-xl group-hover:blur-2xl transition-all duration-500"></div>
                    </div>

                    {/* Info */}
                    <h3 className="text-2xl font-bold text-foreground mb-2">{member.name}</h3>
                    <p className="text-primary font-semibold mb-4">{member.role}</p>
                    <p className="text-muted-foreground mb-6 leading-relaxed">{member.description}</p>

                    {/* Skills */}
                    <div className="mb-6">
                      <p className="text-sm font-medium text-foreground mb-3">Expertise</p>
                      <div className="flex flex-wrap gap-2 justify-center">
                        {member.skills.map((skill) => (
                          <span
                            key={skill}
                            className="px-3 py-1 bg-primary/10 text-primary rounded-full text-xs font-medium"
                          >
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Contact Links */}
                    <div className="flex justify-center space-x-4">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="glass-subtle rounded-full w-10 h-10 p-0 hover:bg-primary/10 hover:text-primary"
                        asChild
                      >
                        <a href={member.github} target="_blank" rel="noopener noreferrer">
                          <Github className="w-4 h-4" />
                        </a>
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="glass-subtle rounded-full w-10 h-10 p-0 hover:bg-primary/10 hover:text-primary"
                        asChild
                      >
                        <a href={member.linkedin} target="_blank" rel="noopener noreferrer">
                          <Linkedin className="w-4 h-4" />
                        </a>
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="glass-subtle rounded-full w-10 h-10 p-0 hover:bg-primary/10 hover:text-primary"
                        asChild
                      >
                        <a href={`mailto:${member.email}`}>
                          <Mail className="w-4 h-4" />
                        </a>
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Technology Stack */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mb-20"
        >
          <Card className="card-premium">
            <CardContent className="p-12">
              <div className="text-center mb-12">
                <h2 className="text-3xl sm:text-4xl font-bold mb-6">Built With Modern Technology</h2>
                <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
                  Our platform leverages cutting-edge technologies to deliver exceptional performance, 
                  security, and user experience.
                </p>
              </div>

              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
                {[
                  { name: "React", description: "Modern UI framework" },
                  { name: "TypeScript", description: "Type-safe development" },
                  { name: "Supabase", description: "Backend & database" },
                  { name: "Tailwind CSS", description: "Utility-first styling" },
                  { name: "Framer Motion", description: "Smooth animations" },
                  { name: "AI/ML APIs", description: "Intelligent analysis" },
                  { name: "PostgreSQL", description: "Robust database" },
                  { name: "Cloud Hosting", description: "Scalable infrastructure" }
                ].map((tech, index) => (
                  <motion.div
                    key={tech.name}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: index * 0.1 }}
                    viewport={{ once: true }}
                    className="text-center group"
                  >
                    <div className="glass-subtle rounded-xl p-6 hover:bg-primary/5 transition-all duration-300 group-hover:scale-105">
                      <h4 className="font-semibold text-foreground mb-2">{tech.name}</h4>
                      <p className="text-sm text-muted-foreground">{tech.description}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Contact CTA */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <Card className="card-premium bg-gradient-glass">
            <CardContent className="p-12">
              <h2 className="text-3xl sm:text-4xl font-bold mb-6">Get In Touch</h2>
              <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
                Have questions about ResumeCheck or want to collaborate? 
                We'd love to hear from you!
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button className="btn-hero">
                  <Mail className="w-4 h-4 mr-2" />
                  Contact Us
                </Button>
                <Button variant="outline" className="btn-glass">
                  <ExternalLink className="w-4 h-4 mr-2" />
                  View on GitHub
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default About;