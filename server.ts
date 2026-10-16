import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';
import fs from 'fs';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json({ limit: '20mb' }));

// Shared Gemini AI client with telemetry header
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY || '',
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// Endpoint: AI Vision Analysis for Campus Cameras & Custom Feeds
app.post('/api/gemini/analyze-vision', async (req, res) => {
  try {
    const { imageBase64, mimeType = 'image/jpeg', cameraLocation, contextNotes } = req.body;

    if (!imageBase64) {
      return res.status(400).json({ error: 'Image data is required.' });
    }

    const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, '');

    const prompt = `You are the Computer Vision & Sensor Diagnostic Engine of an AI-Powered Sustainable Campus Digital Twin.
Location: ${cameraLocation || 'Campus Sensor Node'}
Context Notes: ${contextNotes || 'Routine sustainability surveillance'}

Analyze this campus camera frame with precision.
Inspect for any of the following sustainability defects:
1. Waste accumulation or bin overflow, littering, improper sorting, recyclable contamination.
2. Energy wastage: off-hours illumination in unoccupied rooms, high-bay lighting, excessive HVAC operation, open windows with climate control.
3. Water leakage: leaking joints, dripping valves, pooling water on floors, ponding near irrigation or hydrants.
4. Inefficient resource usage: unoccupied facilities running at full capacity, equipment left idle, solar panel soiling.

Detect bounding boxes for detected objects/issues using normalized coordinates [ymin, xmin, ymax, xmax] between 0 and 1000.
Provide an assessment including:
- primaryIssue (one of: 'WASTE_ACCUMULATION', 'WATER_LEAKAGE', 'ENERGY_WASTAGE', 'RESOURCE_INEFFICIENCY', 'PV_SOILING', 'NOMINAL')
- confidenceScore (0.0 to 1.0)
- severity ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NOMINAL')
- summary (2-3 sentences)
- boundingBoxes: array of objects with box_2d [ymin, xmin, ymax, xmax], label, issueType, confidence
- estimatedWaste: { metric: string, value: number, unit: string, dailyCostUsd: number, annualCo2Kg: number }
- automatedRemediation: immediate automated trigger (e.g. BACnet command, valve closure, work order dispatch)
- facilitiesWorkOrder: { priority: string, department: string, title: string, steps: string[] }
- recommendations: string[] actionable sustainability advice.`;

    if (!process.env.GEMINI_API_KEY) {
      // Fallback structured simulation if key not configured
      return res.json({
        primaryIssue: cameraLocation?.includes('Basement') ? 'WATER_LEAKAGE' :
                      cameraLocation?.includes('Auditorium') ? 'ENERGY_WASTAGE' : 'WASTE_ACCUMULATION',
        confidenceScore: 0.94,
        severity: 'HIGH',
        summary: `Computer vision detected abnormal condition at ${cameraLocation || 'Campus Node'}. Immediate intervention recommended to minimize environmental and financial loss.`,
        boundingBoxes: [
          {
            box_2d: [480, 210, 820, 560],
            label: 'Detected Anomaly Zone',
            issueType: cameraLocation?.includes('Basement') ? 'WATER_LEAKAGE' :
                       cameraLocation?.includes('Auditorium') ? 'ENERGY_WASTAGE' : 'WASTE_ACCUMULATION',
            confidence: 0.95
          }
        ],
        estimatedWaste: {
          metric: 'Resource Loss',
          value: 42.5,
          unit: 'Units/hr',
          dailyCostUsd: 18.50,
          annualCo2Kg: 280
        },
        automatedRemediation: 'Trigger automated supervisory control ticket and notify duty engineer.',
        facilitiesWorkOrder: {
          priority: 'High',
          department: 'Campus Facilities',
          title: `Inspect and resolve issue at ${cameraLocation || 'Campus'}`,
          steps: ['Isolate root cause', 'Apply containment protocol', 'Log resolution in digital twin ledger']
        },
        recommendations: [
          'Deploy schedule-based setback controls to prevent recurrence',
          'Review predictive maintenance interval for this sector'
        ]
      });
    }

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: {
        parts: [
          {
            inlineData: {
              mimeType,
              data: cleanBase64,
            },
          },
          { text: prompt },
        ],
      },
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            primaryIssue: {
              type: Type.STRING,
              description: 'One of: WASTE_ACCUMULATION, WATER_LEAKAGE, ENERGY_WASTAGE, RESOURCE_INEFFICIENCY, PV_SOILING, NOMINAL',
            },
            confidenceScore: { type: Type.NUMBER },
            severity: {
              type: Type.STRING,
              description: 'One of: CRITICAL, HIGH, MEDIUM, LOW, NOMINAL',
            },
            summary: { type: Type.STRING },
            boundingBoxes: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  box_2d: {
                    type: Type.ARRAY,
                    items: { type: Type.INTEGER },
                    description: '[ymin, xmin, ymax, xmax] scaled 0 to 1000',
                  },
                  label: { type: Type.STRING },
                  issueType: { type: Type.STRING },
                  confidence: { type: Type.NUMBER },
                },
                required: ['box_2d', 'label', 'issueType', 'confidence'],
              },
            },
            estimatedWaste: {
              type: Type.OBJECT,
              properties: {
                metric: { type: Type.STRING },
                value: { type: Type.NUMBER },
                unit: { type: Type.STRING },
                dailyCostUsd: { type: Type.NUMBER },
                annualCo2Kg: { type: Type.NUMBER },
              },
              required: ['metric', 'value', 'unit', 'dailyCostUsd', 'annualCo2Kg'],
            },
            automatedRemediation: { type: Type.STRING },
            facilitiesWorkOrder: {
              type: Type.OBJECT,
              properties: {
                priority: { type: Type.STRING },
                department: { type: Type.STRING },
                title: { type: Type.STRING },
                steps: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
              },
              required: ['priority', 'department', 'title', 'steps'],
            },
            recommendations: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
          },
          required: [
            'primaryIssue',
            'confidenceScore',
            'severity',
            'summary',
            'boundingBoxes',
            'estimatedWaste',
            'automatedRemediation',
            'facilitiesWorkOrder',
            'recommendations',
          ],
        },
      },
    });

    const parsed = JSON.parse(response.text || '{}');
    return res.json(parsed);
  } catch (error: any) {
    console.error('Vision analysis error:', error);
    return res.status(500).json({
      error: error.message || 'Computer Vision analysis failed',
    });
  }
});

// Endpoint: AI Actionable Sustainability Recommendations & Dispatch
app.post('/api/gemini/recommendations', async (req, res) => {
  try {
    const { campusTelemetry, activeAlerts } = req.body;

    const prompt = `You are the Lead Sustainability Chief & Autonomous Energy Engineer for a university campus Digital Twin.
Current Campus Metrics:
${JSON.stringify(campusTelemetry, null, 2)}

Active Detected Alerts:
${JSON.stringify(activeAlerts, null, 2)}

Synthesize 4 distinct high-impact actionable interventions spanning:
1. Immediate automated control action (e.g. BACnet HVAC ramp-down, Solenoid valve cutoff, Smart Compactor activation)
2. Operational shift & facilities dispatch
3. Microgrid & solar load balancing
4. Behavioral / Student engagement policy

Return a JSON array of recommendations.`;

    if (!process.env.GEMINI_API_KEY) {
      return res.json({
        recommendations: [
          {
            id: 'REC-01',
            category: 'Energy Wastage',
            title: 'Automated BACnet Lighting & HVAC Setback for Empty Lecture Halls',
            impactScore: 92,
            annualCo2SavingsKg: 14200,
            annualCostSavingsUsd: 4850,
            implementationTime: 'Immediate (Automated Script)',
            actionType: 'BACNET_AUTOMATION',
            description: 'Surveillance confirmed zero occupancy in Main Auditorium while 48kW lighting grid and 100% chilled air are circulating. Send BACnet write command to switch to unoccupied eco-setback mode.',
            steps: [
              'Execute BACnet override command to Main Auditorium lighting controller',
              'Reduce airflow CFM by 60% and adjust thermostat setback by +4°F',
              'Arm passive infrared (PIR) and CV occupancy sensor confirmation'
            ]
          },
          {
            id: 'REC-02',
            category: 'Water Conservation',
            title: 'Isolate Chilled Water Secondary Loop Valve V-102 at Science Hub',
            impactScore: 88,
            annualCo2SavingsKg: 3100,
            annualCostSavingsUsd: 6200,
            implementationTime: '15 Minutes',
            actionType: 'VALVE_ISOLATION',
            description: 'Pressure differential telemetry and basement CV feed indicate a 14 L/min pressurized leak pooling on concrete. Shut valve V-102 and dispatch mechanical plumbing team.',
            steps: [
              'Actuate motorized isolation valve V-102 via SCADA',
              'Generate P1 Emergency Plumbing Work Order with geo-coordinates',
              'Divert secondary loop flow to backup redundant manifold'
            ]
          },
          {
            id: 'REC-03',
            category: 'Waste Management',
            title: 'Dynamic Route Optimization for Campus Electric Compactor Fleet',
            impactScore: 79,
            annualCo2SavingsKg: 5400,
            annualCostSavingsUsd: 3100,
            implementationTime: '1 Hour',
            actionType: 'DISPATCH_WORKORDER',
            description: 'Courtyard solar compactors at 95% capacity with overflow detected by quad camera. Dispatch grounds crew electric cart along optimal route prior to lunch rush.',
            steps: [
              'Transmit route waypoint to Groundskeeping Mobile Terminal',
              'Cycle solar compactor hydraulic compaction cycle remotely',
              'Deploy temporary overflow sorting station near Student Union'
            ]
          },
          {
            id: 'REC-04',
            category: 'Resource Efficiency',
            title: 'Solar Microgrid Peak-Shaving & Thermal Ice Storage Pre-Cooling',
            impactScore: 85,
            annualCo2SavingsKg: 18900,
            annualCostSavingsUsd: 8700,
            implementationTime: 'Automated Schedule',
            actionType: 'MICROGRID_DISPATCH',
            description: 'Predicted solar generation peak between 11:30 and 14:00 exceeds library base load. Pre-cool central library thermal ice tanks during solar crest to prevent tariff peak charges.',
            steps: [
              'Engage 150kWh campus battery energy storage system (BESS) charge cycle',
              'Ramp thermal chiller to 110% using surplus 82kW rooftop solar array',
              'Throttle grid import to baseline during local utility peak window'
            ]
          }
        ]
      });
    }

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            recommendations: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  category: { type: Type.STRING },
                  title: { type: Type.STRING },
                  impactScore: { type: Type.INTEGER },
                  annualCo2SavingsKg: { type: Type.NUMBER },
                  annualCostSavingsUsd: { type: Type.NUMBER },
                  implementationTime: { type: Type.STRING },
                  actionType: { type: Type.STRING },
                  description: { type: Type.STRING },
                  steps: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                },
                required: [
                  'id',
                  'category',
                  'title',
                  'impactScore',
                  'annualCo2SavingsKg',
                  'annualCostSavingsUsd',
                  'implementationTime',
                  'actionType',
                  'description',
                  'steps',
                ],
              },
            },
          },
          required: ['recommendations'],
        },
      },
    });

    const parsed = JSON.parse(response.text || '{}');
    return res.json(parsed);
  } catch (error: any) {
    console.error('Recommendations error:', error);
    return res.status(500).json({ error: error.message || 'Recommendation generation failed' });
  }
});

// Endpoint: AI Sustainability & AASHE STARS Report Generator
app.post('/api/gemini/generate-report', async (req, res) => {
  try {
    const { reportType = 'AASHE_STARS_AUDIT', campusMetrics, resolvedIncidents, activeIncidents } = req.body;

    const prompt = `You are the Lead Environmental Scientist & Sustainability Auditor for an institution of higher education.
Generate a comprehensive, publication-ready sustainability report based on this real-time campus Digital Twin data:
- Report Type: ${reportType}
- Metrics: ${JSON.stringify(campusMetrics, null, 2)}
- Active Issues: ${JSON.stringify(activeIncidents, null, 2)}
- Resolved Actions: ${JSON.stringify(resolvedIncidents, null, 2)}

Create an exhaustive, structured report following standard institutional ESG and AASHE STARS (Sustainability Tracking, Assessment & Rating System) criteria.
Include:
1. Executive Summary
2. Quantitative Performance Ledger (Energy, Water, Waste, Carbon)
3. Computer Vision & IoT Anomaly Audit
4. Financial Impact & Cost-Benefit Analysis
5. AASHE STARS Rating Assessment (Targeting Platinum)
6. 5-Year Net-Zero Strategic Roadmap`;

    if (!process.env.GEMINI_API_KEY) {
      return res.json({
        title: 'Comprehensive Campus Sustainability & Net-Zero Operations Audit',
        institution: 'University Digital Twin Laboratory',
        ratingLevel: 'STARS Gold (81.4 Points) — Platinum Path Viable',
        date: new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }),
        executiveSummary: 'This audit summarizes real-time performance data captured by the Campus Digital Twin, incorporating real-time computer vision surveillance and IoT sensor telemetry. Total campus energy consumption was maintained at 412 kW with 38% solar self-generation. Three critical resource inefficiencies were identified and remediated autonomously, preventing an estimated 28,600 kg of annual CO2 emissions and $14,150 in utility loss.',
        kpiSummary: [
          { kpi: 'Net Energy Use Intensity (EUI)', value: '64.2 kBtu/sq.ft/yr', benchmark: '78.0 kBtu/sq.ft/yr', status: 'Exceeding Target' },
          { kpi: 'On-Site Renewable Fraction', value: '38.4%', benchmark: '30.0%', status: 'Exceeding Target' },
          { kpi: 'Water Consumption per Capita', value: '28.6 L/student-day', benchmark: '35.0 L/student-day', status: 'On Track' },
          { kpi: 'Waste Diversion Rate', value: '71.8%', benchmark: '80.0%', status: 'Requires Intervention' },
          { kpi: 'Building Occupancy Efficiency', value: '84.2%', benchmark: '75.0%', status: 'Optimal' },
        ],
        cvAnomalyHighlights: [
          'Courtyard Station: Computer vision identified bin overflow at 14:35; automated routing dispatched electric custodial hauler within 12 minutes, diverting 45kg from landfill.',
          'Science Hub Basement: Subsurface camera detected 14 L/min pipe weeping; pressure differential telemetry triggered automated valve isolation before hydraulic damage occurred.',
          'Main Auditorium: Off-hours vision sweep flagged 48kW luminaire grid active in empty hall; automated BACnet command instituted immediate 90% lumen setback.'
        ],
        financialAnalysis: {
          annualUtilityLossMitigatedUsd: 34200,
          projectedPaybackPeriodYears: 1.4,
          carbonOffsetValueUsd: 4800,
          operationalCostReductionPercent: 14.8
        },
        strategicRoadmap: [
          { year: 'Phase 1 (Q1-Q2)', milestone: 'Expand CV cameras to student dormitories and parking structure EV hubs' },
          { year: 'Phase 2 (Q3-Q4)', milestone: 'Commission 250kW rooftop bifacial PV arrays on Engineering Complex' },
          { year: 'Phase 3 (Next Year)', milestone: 'Integrate campus greywater recycling network with AI soil moisture irrigation' },
          { year: 'Phase 4 (Year 3-5)', milestone: 'Achieve AASHE STARS Platinum and net-zero operational carbon emissions' }
        ],
        concludingRemarks: 'The integration of high-resolution computer vision with physical building management systems creates an autonomous self-healing campus infrastructure that lowers operational expenditure while accelerating climate commitment milestones.'
      });
    }

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            institution: { type: Type.STRING },
            ratingLevel: { type: Type.STRING },
            date: { type: Type.STRING },
            executiveSummary: { type: Type.STRING },
            kpiSummary: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  kpi: { type: Type.STRING },
                  value: { type: Type.STRING },
                  benchmark: { type: Type.STRING },
                  status: { type: Type.STRING },
                },
                required: ['kpi', 'value', 'benchmark', 'status'],
              },
            },
            cvAnomalyHighlights: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            financialAnalysis: {
              type: Type.OBJECT,
              properties: {
                annualUtilityLossMitigatedUsd: { type: Type.NUMBER },
                projectedPaybackPeriodYears: { type: Type.NUMBER },
                carbonOffsetValueUsd: { type: Type.NUMBER },
                operationalCostReductionPercent: { type: Type.NUMBER },
              },
              required: [
                'annualUtilityLossMitigatedUsd',
                'projectedPaybackPeriodYears',
                'carbonOffsetValueUsd',
                'operationalCostReductionPercent',
              ],
            },
            strategicRoadmap: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  year: { type: Type.STRING },
                  milestone: { type: Type.STRING },
                },
                required: ['year', 'milestone'],
              },
            },
            concludingRemarks: { type: Type.STRING },
          },
          required: [
            'title',
            'institution',
            'ratingLevel',
            'date',
            'executiveSummary',
            'kpiSummary',
            'cvAnomalyHighlights',
            'financialAnalysis',
            'strategicRoadmap',
            'concludingRemarks',
          ],
        },
      },
    });

    const parsed = JSON.parse(response.text || '{}');
    return res.json(parsed);
  } catch (error: any) {
    console.error('Report generation error:', error);
    return res.status(500).json({ error: error.message || 'Report generation failed' });
  }
});

// Static image helper for preloaded CCTV frames
app.get('/api/sample-image/:filename', (req, res) => {
  const filename = req.params.filename;
  const imagePath = path.join(__dirname, 'src', 'assets', 'images', filename);
  if (fs.existsSync(imagePath)) {
    return res.sendFile(imagePath);
  }
  return res.status(404).send('Not found');
});

// Vite middleware in dev or static files in production
async function startServer() {
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Campus Digital Twin server running at http://0.0.0.0:${PORT}`);
  });
}

startServer();
