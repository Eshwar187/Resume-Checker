import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ArrowLeft, Shield, Eye, Database, Lock, Cookie } from "lucide-react";

const Privacy = () => {
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
            Privacy Policy
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            How we collect, use, and protect your personal information
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
                <Shield className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">1. Information We Collect</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                We collect information you provide directly to us, such as when you create an account, 
                upload resumes, or contact us for support.
              </p>
              <ul className="list-disc list-inside text-muted-foreground ml-4 space-y-2">
                <li><strong>Personal Information:</strong> Name, email address, password</li>
                <li><strong>Resume Data:</strong> Resume files, job descriptions, and analysis results</li>
                <li><strong>Usage Information:</strong> How you interact with our service</li>
                <li><strong>Device Information:</strong> Browser type, operating system, IP address</li>
              </ul>
            </div>

            {/* Section 2 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Eye className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">2. How We Use Your Information</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                We use the information we collect for various purposes:
              </p>
              <ul className="list-disc list-inside text-muted-foreground ml-4 space-y-2">
                <li>Provide, maintain, and improve our service</li>
                <li>Process and analyze your resumes and job descriptions</li>
                <li>Send you technical notices and support messages</li>
                <li>Respond to your comments and questions</li>
                <li>Monitor and analyze trends and usage</li>
              </ul>
            </div>

            {/* Section 3 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Database className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">3. Information Sharing</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                We do not sell, trade, or otherwise transfer your personal information to third parties. 
                This does not include trusted third parties who assist us in operating our website, 
                conducting our business, or servicing you, so long as those parties agree to keep this information confidential.
              </p>
            </div>

            {/* Section 4 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Lock className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">4. Data Security</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                We implement a variety of security measures to maintain the safety of your personal information:
              </p>
              <ul className="list-disc list-inside text-muted-foreground ml-4 space-y-2">
                <li>Encryption of sensitive data in transit and at rest</li>
                <li>Regular security assessments and updates</li>
                <li>Access controls and authentication measures</li>
                <li>Secure cloud infrastructure with Supabase</li>
              </ul>
            </div>

            {/* Section 5 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Cookie className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">5. Cookies and Tracking</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                We use cookies and similar tracking technologies to track activity on our service and hold certain information. 
                You can instruct your browser to refuse all cookies or to indicate when a cookie is being sent.
              </p>
            </div>

            {/* Section 6 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Shield className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">6. Your Rights</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                You have certain rights regarding your personal information:
              </p>
              <ul className="list-disc list-inside text-muted-foreground ml-4 space-y-2">
                <li>Access and update your personal information</li>
                <li>Delete your account and associated data</li>
                <li>Export your data in a portable format</li>
                <li>Opt-out of certain communications</li>
              </ul>
            </div>

            {/* Section 7 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Database className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">7. Data Retention</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                We retain your personal information only for as long as necessary to provide you with our service 
                and as described in this Privacy Policy. We will retain and use your information to the extent 
                necessary to comply with our legal obligations, resolve disputes, and enforce our policies.
              </p>
            </div>

            {/* Section 8 */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Eye className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-semibold text-foreground">8. Changes to This Policy</h2>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                We may update our Privacy Policy from time to time. We will notify you of any changes by posting 
                the new Privacy Policy on this page and updating the "Last updated" date.
              </p>
            </div>

            {/* Contact */}
            <div className="border-t border-border pt-8">
              <h3 className="text-lg font-semibold text-foreground mb-4">Contact Information</h3>
              <p className="text-muted-foreground">
                If you have any questions about this Privacy Policy, please contact us at{" "}
                <a href="mailto:privacy@resumecheck.com" className="text-primary hover:text-primary-glow">
                  privacy@resumecheck.com
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

export default Privacy;