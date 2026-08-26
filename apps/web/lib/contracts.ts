/**
 * Re-export of the generated contracts.
 *
 * Every component imports types from here rather than reaching into
 * `packages/contracts/dist` directly. One indirection, two payoffs: the deep
 * relative path exists in exactly one file, and the import site reads as a
 * dependency on the contract rather than on a build artifact.
 *
 * These types are generated from the pydantic models. Do not hand-edit the
 * generated file — change the Python models and re-run `pnpm codegen`.
 */

export type {
  Anchor,
  AnchorStatus,
  AnswerBlock,
  AnswerStatus,
  BBox,
  ChoiceGroup,
  DocumentKind,
  GradeResult,
  Highlight,
  InkRegion,
  Line,
  LineIndex,
  Mapping,
  MappingResult,
  MatchEvidence,
  MatchSignal,
  NumberingGap,
  OcrEngine,
  OrphanAnswer,
  Page,
  PageBox,
  ProgressEvent,
  Question,
  QuestionGrade,
  QuestionPaper,
  Requirement,
  RubricPoint,
  Section,
  SourceFile,
  Stage,
  Stem,
  Submission,
  Word,
} from "../../../packages/contracts/dist/types";
