require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
app.use(cors());
app.use(express.json());
// === NEW: Health Check Route ===
app.get('/', (req, res) => {
    res.send("✅ Node.js MERN Backend is awake and ready for React!");
});
// 1. Initialize the Gemini AI SDK
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.post('/api/evaluate-patient', async (req, res) => {
    try {
        const patientData = req.body;
        console.log("1. Received patient vitals from React.");

        // 2. Fetch the deterministic risk score and classification from Python
        const flaskResponse = await axios.post('http://127.0.0.1:5000/analyze', patientData);
        const aiData = flaskResponse.data;
        console.log("2. Received clinical rules from Python AI.");

        // 3. Extract the variables to feed into the LLM
        const riskPercentage = aiData.ai_metrics.risk_percentage;
        const hfType = aiData.clinical_report.hf_type;
        const nyhaClass = aiData.clinical_report.nyha_class;
        const stage = aiData.clinical_report.stage;
        const suggestion = aiData.clinical_report.suggestion;

        // 4. THE LLM BRIDGE: Construct the prompt for the generative model
        console.log("3. Generating doctor's narrative with Gemini...");
        const prompt = `
        You are a highly professional cardiology AI assistant. I will provide you with the raw output from our Multimodal Heart Failure Neural Network. 
        
        Patient AI Risk Score: ${riskPercentage}% mortality/event risk.
        Classification: ${hfType}
        Severity: ${nyhaClass}, ${stage}
        System Recommended Protocol: ${suggestion}
        
        Task: Write a concise, 3-paragraph clinical summary for the attending physician. 
        - Paragraph 1: State the diagnosis and AI risk severity clearly.
        - Paragraph 2: Explain the physiological mechanism of this specific classification.
        - Paragraph 3: Detail the recommended clinical action plan (GDMT, devices, etc.).
        
        Tone: Clinical, objective, and urgent if the risk score is high. Do not use markdown formatting like asterisks, just return plain text.
        `;

        // 5. Call the Gemini API
        const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
        const result = await model.generateContent(prompt);
        const doctorNarrative = result.response.text();
        console.log("4. Narrative generated successfully.");

        // 6. Send the ultimate payload back to React
        res.status(200).json({
            success: true,
            payload: {
                raw_metrics: aiData.ai_metrics,
                clinical_rules: aiData.clinical_report,
                generated_narrative: doctorNarrative // React will display this paragraph!
            }
        });

    } catch (error) {
        console.error("❌ Backend Error:", error.message);
        res.status(500).json({ 
            success: false, 
            message: "Failed to process the patient evaluation." 
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`✅ Node.js MERN Backend running on http://localhost:${PORT}`);
});