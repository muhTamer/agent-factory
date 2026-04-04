import { SetupWizard } from "@/components/setup/SetupWizard";
import { AuthGate } from "@/components/AuthGate";

export default function Home() {
  return (
    <AuthGate>
      <SetupWizard />
    </AuthGate>
  );
}
