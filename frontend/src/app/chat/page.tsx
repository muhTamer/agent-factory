import { ChatContainer } from "@/components/chat/ChatContainer";
import { AuthGate } from "@/components/AuthGate";

export default function ChatPage() {
  return (
    <AuthGate>
      <ChatContainer />
    </AuthGate>
  );
}
