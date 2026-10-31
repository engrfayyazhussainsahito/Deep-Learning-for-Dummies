"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = Home;
var react_1 = require("react");
var Layout_1 = require("@theme/Layout");
var Link_1 = require("@docusaurus/Link");
function Home() {
    return (<Layout_1.default>
      <div style={{
            padding: '100px 20px',
            maxWidth: '800px',
            margin: '0 auto',
            textAlign: 'center',
            fontFamily: 'var(--ifm-font-family-base)',
        }}>
        <h1 style={{
            fontSize: '3.5rem',
            fontWeight: '800',
            letterSpacing: '-0.04em',
            marginBottom: '2rem',
            lineHeight: '1.1',
        }}>
          Deep Learning <br />
          <span style={{ color: 'var(--ifm-color-primary)' }}>Academy</span>
        </h1>

        <p style={{
            fontSize: '1.5rem',
            lineHeight: '1.6',
            color: '#666',
            marginBottom: '3rem',
            fontFamily: 'var(--ifm-font-family-text)',
        }}>
          A rigorous, 18-week journey from mathematical foundations to the frontier of generative AI.
          Designed for those who seek depth over breadth.
        </p>

        <div style={{
            display: 'flex',
            gap: '20px',
            justifyContent: 'center',
        }}>
          <Link_1.default to="/docs/intro" style={{
            backgroundColor: '#000',
            color: '#fff',
            padding: '15px 30px',
            borderRadius: '5px',
            fontWeight: '600',
            textDecoration: 'none',
            fontSize: '1.1rem',
        }}>
            Begin the Journey
          </Link_1.default>
          <Link_1.default to="/docs/week-01-foundations/lecture-1" style={{
            border: '1px solid #ddd',
            color: '#000',
            padding: '15px 30px',
            borderRadius: '5px',
            fontWeight: '600',
            textDecoration: 'none',
            fontSize: '1.1rem',
        }}>
            Explore Curriculum
          </Link_1.default>
        </div>

        <div style={{
            marginTop: '100px',
            textAlign: 'left',
            borderTop: '1px solid #eee',
            paddingTop: '50px',
            fontSize: '1rem',
            color: '#888',
            fontFamily: 'var(--ifm-font-family-base)',
        }}>
          <p><strong>Curriculum standard:</strong> Inspired by Stanford CS231n, MIT 6.S191, and the Goodfellow et al. textbook.</p>
          <p><strong>Focus:</strong> First-principles derivation, modular PyTorch architecture, and academic rigor.</p>
        </div>
      </div>
    </Layout_1.default>);
}
