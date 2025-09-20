import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ArrowLeft, Shield, FileText, Users, AlertTriangle } from "lucide-react";

const Terms = () => {
  return (
    <div className="min-h-screen px-4 sm:px-6 lg:px-8 pt-20 pb-16">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl sm:text-5xl font-bold text-foreground mb-4">
            Terms of Service
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Your rights and responsibilities when using ResumeCheck
          </p>
        </motion.div>

        {/* Content */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="card-premium"
        >
          <div className="space-y-8">
            {/* Section 1 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <FileText className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">1. Acceptance of Terms</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                By accessing and using ResumeCheck, you accept and agree to be bound by the terms and provision of this agreement. 
                If you do not agree to abide by the above, please do not use this service.
              </p>
            </div>

            {/* Section 2 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Users className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">2. Use License</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                Permission is granted to temporarily download one copy of ResumeCheck per device for personal, 
                non-commercial transitory viewing only. This is the grant of a license, not a transfer of title, and under this license you may not:
              </p>
              <ul className="list-disc list-inside text-muted-foreground ml-4 space-y-2">
                <li>modify or copy the materials</li>
                <li>use the materials for any commercial purpose or for any public display</li>
                <li>attempt to reverse engineer any software contained on the website</li>
                <li>remove any copyright or other proprietary notations from the materials</li>
              </ul>
            </div>

            {/* Section 3 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Shield className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">3. Privacy Policy</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                Your privacy is important to us. Our Privacy Policy explains how we collect, use, and protect your information when you use our service. 
                By using ResumeCheck, you agree to the collection and use of information in accordance with our Privacy Policy.
              </p>
            </div>

            {/* Section 4 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <AlertTriangle className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">4. Disclaimer</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                The materials on ResumeCheck are provided on an 'as is' basis. ResumeCheck makes no warranties, 
                expressed or implied, and hereby disclaims and negates all other warranties including without limitation, 
                implied warranties or conditions of merchantability, fitness for a particular purpose, or non-infringement of intellectual property or other violation of rights.
              </p>
            </div>

            {/* Section 5 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <FileText className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">5. Limitations</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                In no event shall ResumeCheck or its suppliers be liable for any damages (including, without limitation, 
                damages for loss of data or profit, or due to business interruption) arising out of the use or inability to use 
                the materials on ResumeCheck, even if ResumeCheck or a ResumeCheck authorized representative has been notified orally or in writing of the possibility of such damage.
              </p>
            </div>

            {/* Section 6 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Users className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">6. Revisions and Errata</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                The materials appearing on ResumeCheck could include technical, typographical, or photographic errors. 
                ResumeCheck does not warrant that any of the materials on its website are accurate, complete, or current. 
                ResumeCheck may make changes to the materials contained on its website at any time without notice.
              </p>
            </div>

            {/* Contact */}
            <div className="border-t border-border pt-8">
              <h3 className="text-lg font-semibold text-foreground mb-4">Contact Information</h3>
              <p className="text-muted-foreground">
                If you have any questions about these Terms of Service, please contact us at{" "}
                <a href="mailto:support@resumecheck.com" className="text-primary hover:text-primary-glow">
                  support@resumecheck.com
                </a>
              </p>
              <p className="text-sm text-muted-foreground mt-4">
                Last updated: {new Date().toLocaleDateString()}
              </p>
            </div>
          </div>
        </motion.div>

        {/* Back Navigation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-8 text-center"
        >
          <Link
            to="/signup"
            className="inline-flex items-center text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Sign Up
          </Link>
        </motion.div>
      </div>
    </div>
  );
};

export default Terms;