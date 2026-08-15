export const CATEGORY_NAMES = {
  template: 'Template',
  warmup: 'Warm Up',
  praise: 'Praise',
  suggest: 'Suggest',
  listen: 'Listen',
  direct: 'Direct',
};

// Standardized to Purple family according to Visual Design System (Priority 7)
export const CATEGORY_COLORS = {
  template: '#8b5cf6',
  warmup: '#a78bfa',
  praise: '#7c3aed',
  suggest: '#6d28d9',
  listen: '#5c21df',
  direct: '#8b5cf6',
};

export const getGuidelineDetails = (cat, results = []) => {
  const name = cat.name;
  const score = cat.score;
  const max = cat.max_score;
  const desc = cat.description || "";
  
  let title = name.charAt(0).toUpperCase() + name.slice(1);
  let explanation = "";
  let evidence = [];
  let positiveExamples = [];
  let missedOpportunities = [];
  let recommendation = "";

  if (name === "template") {
    title = CATEGORY_NAMES.template;
    explanation = "Verifies the sequence and completion of standard conversation phases (Warm Up, Praise, Suggestions, Listening, Directives).";
  } else if (name === "warmup") {
    title = CATEGORY_NAMES.warmup;
    explanation = "Evaluates the initial greeting, rapport building, and tone of the opening segments.";
  } else if (name === "praise") {
    title = CATEGORY_NAMES.praise;
    explanation = "Assesses the presence and warmth of validation, positive feedback, and supportive responses.";
  } else if (name === "suggest") {
    title = CATEGORY_NAMES.suggest;
    explanation = "Analyzes the constructive suggestions, feedback delivery, and the balance of positive vs. negative framing.";
  } else if (name === "listen") {
    title = CATEGORY_NAMES.listen;
    explanation = "Measures active listening segments, pauses, and backchannel responses to ensure dialogue rather than monologue.";
  } else if (name === "direct") {
    title = CATEGORY_NAMES.direct;
    explanation = "Checks if action items, goals, and direct instructions are clear, realistic, and constructive.";
  }

  const matchingSegs = results.filter(s => {
    if (name === "template") return false;
    if (name === "warmup") return s.template_label === "WarmUp";
    if (name === "praise") return s.template_label === "Praise";
    if (name === "suggest") return s.template_label === "PSuggest" || s.template_label === "NSuggest";
    if (name === "listen") return s.template_label === "Listen";
    if (name === "direct") return s.template_label === "Direct";
    return false;
  });

  matchingSegs.slice(0, 2).forEach(s => {
    positiveExamples.push(`"${s.text}" (${s.speaker})`);
  });

  if (name === "template") {
    const completedMatch = desc.match(/completed categories - \[(.*?)\]/);
    const missedMatch = desc.match(/missed categories - \[(.*?)\]/);
    
    if (completedMatch && completedMatch[1]) {
      evidence.push(`Completed phases: ${completedMatch[1]}`);
    }
    if (missedMatch && missedMatch[1]) {
      missedOpportunities.push(`Missing/Out-of-order phases: ${missedMatch[1]}`);
    }
    if (desc.includes("WarmUp was not at the beginning")) {
      missedOpportunities.push("WarmUp phase occurred late or was preceded by other topics.");
    }
    recommendation = "Ensure all 5 core phases (WarmUp, Praise, Suggest, Listen, Direct) are covered in the correct chronological order.";
  } else {
    if (desc.includes("Good speaking tone maintained")) {
      evidence.push("Appropriate tone maintained during this phase.");
    } else if (desc.includes("Tone must improve") || desc.includes("Tone should be improved")) {
      missedOpportunities.push("Speaking tone was evaluated as non-optimal (sad, flat or aggressive).");
    }

    const suggestionIdx = desc.indexOf("Suggestions:");
    const recommendationIdx = desc.indexOf("Recommendations not followed:");
    if (suggestionIdx !== -1) {
      recommendation = desc.substring(suggestionIdx + 12).trim();
    } else if (recommendationIdx !== -1) {
      recommendation = desc.substring(recommendationIdx + 29).trim();
    } else {
      if (name === "warmup") recommendation = "Spend another 20–30 seconds building rapport before moving into evaluation.";
      else if (name === "praise") recommendation = "Actively validate contributions and use positive reinforcement before highlighting gaps.";
      else if (name === "suggest") recommendation = "Ensure suggestions are constructively framed (aim for at least 30% positive suggestions).";
      else if (name === "listen") recommendation = "Invite feedback, ask open-ended questions, and pause to let the other speaker respond.";
      else if (name === "direct") recommendation = "Keep directives clear, concise, and focused on specific action items.";
    }

    const baseEvidence = desc.split(".")[0];
    if (baseEvidence && !baseEvidence.startsWith("Suggestions") && !baseEvidence.startsWith("Recommendations")) {
      evidence.push(baseEvidence + ".");
    }
  }

  if (!recommendation || recommendation.includes("RAG evaluation failed")) {
    if (name === "warmup") recommendation = "Greet participants warmly by name and check in on their week before jumping to business.";
    else if (name === "praise") recommendation = "Praise team members' work explicitly and highlight specific accomplishments.";
    else if (name === "suggest") recommendation = "Frame constructive criticism with actionable steps and positive encouragement.";
    else if (name === "listen") recommendation = "Ask 'What are your thoughts on this?' to encourage passive speakers to speak.";
    else if (name === "direct") recommendation = "Wrap up the session by summarizing clear next steps and owners.";
  }

  if (evidence.length === 0) {
    evidence.push("No direct evidence logged.");
  }
  if (missedOpportunities.length === 0 && score < max) {
    if (name === "warmup") missedOpportunities.push("Missed establishing deep personal rapport after initial greetings.");
    else if (name === "praise") missedOpportunities.push("Validation segments were absent or lacked warm emotional resonance.");
    else if (name === "suggest") missedOpportunities.push("Feedback was heavily weighted towards critique without balanced praise.");
    else if (name === "listen") missedOpportunities.push("Active listening/backchannel indicator below the 10% threshold.");
    else if (name === "direct") missedOpportunities.push("Action items and directives were not explicitly outlined or summarized.");
  }

  return {
    title,
    explanation,
    evidence,
    positiveExamples,
    missedOpportunities,
    recommendation
  };
};

export const getScoreStatus = (score, max) => {
  if (max <= 0) return { label: "N/A", color: "var(--text-muted)", type: "none" };
  const ratio = score / max;
  if (ratio >= 0.85) return { label: "Excellent", color: "#8b5cf6", type: "excellent" }; // Standardized to purple status color
  if (ratio >= 0.70) return { label: "Good", color: "#a78bfa", type: "good" };
  if (ratio >= 0.50) return { label: "Needs Improvement", color: "var(--warning)", type: "warning" };
  return { label: "Poor", color: "var(--danger)", type: "danger" };
};
