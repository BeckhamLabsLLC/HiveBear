import { useEffect, useState } from "react";
import type { HardwareProfile, ModelRecommendation } from "../types";
import { getHardwareProfile, getRecommendations } from "../lib/invoke";

export function useProfile() {
  const [profile, setProfile] = useState<HardwareProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHardwareProfile()
      .then(setProfile)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return { profile, loading, error };
}

export function useRecommendations() {
  const [recommendations, setRecommendations] = useState<ModelRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getRecommendations()
      .then(setRecommendations)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return { recommendations, loading, error };
}
