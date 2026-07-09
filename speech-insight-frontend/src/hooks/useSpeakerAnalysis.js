import { useMemo } from 'react';
import { getSpeakerTheme, mapRoleLabel } from '../utils/speakerUtils';

export const useSpeakerAnalysis = (metadata, results = [], report = null) => {
  const speakersList = useMemo(() => {
    if (!results || !Array.isArray(results)) return [];
    return Array.from(new Set(results.map(r => r.speaker)));
  }, [results]);

  const totalDuration = metadata?.total_duration || 1;

  // Memoize analysis data for efficiency
  const speakerAnalysisData = useMemo(() => {
    if (!metadata || !results || results.length === 0) return [];

    const speakerRoles = report?.speaker_stats || metadata.speaker_roles || {};
    const leadSpeaker = report?.lead_speaker || metadata.lead_speaker;

    return speakersList.map(speaker => {
      const segments = results.filter(r => r.speaker === speaker);
      const talkTime = segments.reduce((acc, r) => acc + (r.end_time - r.start_time), 0);
      const percentage = totalDuration > 0 ? (talkTime / totalDuration) * 100 : 0;
      const turns = segments.length;
      const avgDuration = turns > 0 ? talkTime / turns : 0;

      // Extract behaviors
      let questions = 0;
      let suggestions = 0;
      let praise = 0;
      let listening = 0;
      let direct = 0;
      let warmup = 0;

      const emotions = {};
      let totalSentiment = 0;
      let sentimentCount = 0;

      segments.forEach(seg => {
        if (seg.text && seg.text.includes("?")) {
          questions++;
        }

        const tLabel = seg.template_label || "";
        if (tLabel === "NSuggest" || tLabel === "PSuggest") {
          suggestions++;
        } else if (tLabel === "Praise") {
          praise++;
        } else if (tLabel === "Listen") {
          listening++;
        } else if (tLabel === "Direct") {
          direct++;
        } else if (tLabel === "WarmUp") {
          warmup++;
        }

        // Parse emotion label
        const emoStr = seg.emotion || "neutral";
        const emo = emoStr.split(' ')[0].toLowerCase();
        emotions[emo] = (emotions[emo] || 0) + 1;

        if (seg.vader && typeof seg.vader.compound === "number") {
          totalSentiment += seg.vader.compound;
          sentimentCount++;
        }
      });

      let dominantEmotion = "neutral";
      let maxEmoCount = 0;
      Object.entries(emotions).forEach(([emo, count]) => {
        if (count > maxEmoCount) {
          maxEmoCount = count;
          dominantEmotion = emo;
        }
      });

      const avgSentiment = sentimentCount > 0 ? totalSentiment / sentimentCount : 0;
      const roleInfo = speakerRoles[speaker] || {};
      
      // Enforce the Leader presentation mapping at analysis data preparation stage
      const predictedRole = roleInfo.role ? mapRoleLabel(roleInfo.role) : "Other";
      
      const confidence = typeof roleInfo.confidence === "number" ? roleInfo.confidence :
                         typeof roleInfo.probability === "number" ? roleInfo.probability : 0.0;
                         
      const evidence = roleInfo.evidence || [];
      const theme = getSpeakerTheme(speaker, speakersList);

      // Dynamically compute prediction factors / behaviour summary from statistics
      const predictionFactors = [];
      if (speaker === leadSpeaker) {
        predictionFactors.push("High speaking duration / initiates structure");
      }
      if (percentage > 35) {
        predictionFactors.push("Conversational dominance");
      }
      if (praise > 0) {
        predictionFactors.push("Uses appraisal / positive feedback vocabulary");
      }
      if (direct > 0) {
        predictionFactors.push("Uses directive or instructive language");
      }
      if (suggestions > 0) {
        predictionFactors.push("Gives constructive proposals and suggestions");
      }
      if (questions > 0) {
        predictionFactors.push("Frequently asks questions / guides dialogue");
      }
      if (listening > 0) {
        predictionFactors.push("Provides active listening / agreement feedback");
      }

      if (predictionFactors.length === 0) {
        predictionFactors.push("General participation and balanced discussion contribution");
      }

      return {
        name: speaker,
        isLead: speaker === leadSpeaker,
        predictedRole,
        confidence,
        talkTime: talkTime.toFixed(1),
        percentage: percentage.toFixed(0),
        turns,
        avgDuration: avgDuration.toFixed(1),
        avgSentiment: avgSentiment.toFixed(2),
        dominantEmotion: dominantEmotion.charAt(0).toUpperCase() + dominantEmotion.slice(1).toLowerCase(),
        evidence,
        theme,
        predictionFactors,
        behavior: {
          questions,
          suggestions,
          praise,
          listening,
          direct,
          warmup
        },
        xgboost: roleInfo.xgboost,
        gemini: roleInfo.gemini,
        finalRole: roleInfo.final_role || roleInfo.role,
        predictionSource: roleInfo.prediction_source || (roleInfo.prediction_details && roleInfo.prediction_details.source) || "xgboost"
      };
    });
  }, [metadata, results, speakersList, totalDuration, report]);

  const compositionStats = useMemo(() => {
    const stats = { Leader: 0, HR: 0, Junior: 0, Other: 0 };
    speakerAnalysisData.forEach(spk => {
      const roleLower = spk.predictedRole.toLowerCase();
      if (roleLower === "leader" || roleLower === "manager") stats.Leader++;
      else if (roleLower === "hr") stats.HR++;
      else if (roleLower === "junior") stats.Junior++;
      else stats.Other++;
    });
    return stats;
  }, [speakerAnalysisData]);

  return {
    speakersList,
    speakerAnalysisData,
    compositionStats,
    totalDuration
  };
};
