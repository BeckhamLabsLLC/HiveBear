import { useEffect, useState } from "react";
import type { HardwareProfile, ModelRecommendation } from "../types";
import { getHardwareProfile, getRecommendations } from "../lib/invoke";
import { useHasOnboarded } from "../components/WelcomeModal";

export function useProfile() {
  const ready = useHasOnboarded();
  const [profile, setProfile] = useState<HardwareProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ready) return;
    setLoading(true);
    getHardwareProfile()
      .then(setProfile)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [ready]);

  return { profile, loading, error };
}

export function useRecommendations() {
  const ready = useHasOnboarded();
  const [recommendations, setRecommendations] = useState<ModelRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ready) return;
    setLoading(true);
    getRecommendations()
      .then(setRecommendations)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [ready]);

  return { recommendations, loading, error };
}
