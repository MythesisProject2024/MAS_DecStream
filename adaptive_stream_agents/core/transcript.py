from __future__ import annotations

from dataclasses import dataclass, field

from adaptive_stream_agents.core.domain import NegotiationMessage


@dataclass
class Transcript:
    messages: list[NegotiationMessage] = field(default_factory=list)

    def add(self, message: NegotiationMessage) -> None:
        self.messages.append(message)

    def extend(self, messages: list[NegotiationMessage]) -> None:
        self.messages.extend(messages)

    def to_text(self) -> str:
        lines = ["Adaptive stream scheduling negotiation", "=" * 47]
        current_round: int | None = None
        for msg in self.messages:
            if msg.round_id != current_round:
                current_round = msg.round_id
                lines.append("")
                lines.append(f"Round {current_round}: {msg.intent}")
                lines.append("-" * (len(lines[-1])))
            lines.append(f"{msg.sender} -> {msg.receiver}: {msg.content}")
        return "\n".join(lines)

    def to_latex(self) -> str:
        blocks: list[str] = []
        for msg in self.messages:
            align = "flushleft" if msg.sender == "A0" else "flushright"
            box = "leftmsg" if msg.sender == "A0" else "rightmsg"
            safe_content = msg.content.replace("&", r"\&").replace("%", r"\%")
            blocks.append(
                "\\begin{"
                + align
                + "}\n"
                + "\\begin{tcolorbox}["
                + box
                + "]\n"
                + f"\\textbf{{Round {msg.round_id}: {msg.intent}}}\\\\\n"
                + f"\\faRobot\\ \\textbf{{Agent ${msg.sender}$}}\\\\\n"
                + safe_content.replace("\n", "\\\\\n")
                + "\n\\end{tcolorbox}\n"
                + "\\end{"
                + align
                + "}"
            )
        return "\n\n".join(blocks)

