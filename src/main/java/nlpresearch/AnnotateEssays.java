package nlpresearch;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

public class AnnotateEssays {

    public static final String DELIM = "|||";

    public static class Tuple<X, Y> {
        public final X x;
        public final Y y;
        public Tuple(X x, Y y) {
            this.x = x;
            this.y = y;
        }
    }

    private static List<String> findFiles(String folder){

        List<String> results = new ArrayList<String>();
        File[] files = new File(folder).listFiles();
        //If this pathname does not denote a directory, then listFiles() returns null.

        for (File file : files) {
            if (file.isFile() && file.getName().endsWith(".txt")) {
                results.add(file.getAbsolutePath());
            }
        }
        return results;
    }

    private static String readFile(String fname) throws IOException {
        StringBuilder sb = new StringBuilder();
        BufferedReader br = new BufferedReader(new FileReader(fname));
        try {

            String line = br.readLine();

            while (line != null) {
                sb.append(line);
                sb.append(System.lineSeparator());
                line = br.readLine();
            }
            String everything = sb.toString();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            br.close();
        }
        return sb.toString();
    }

    private static List<List<String>> copySentences(List<List<String>> sentences){
        List<List<String>> copySents = new ArrayList<>();
        for(List<String> sent: sentences){
            List<String> copySent = copyWithoutEmptyStrings(sent);
            copySents.add(copySent);
        }
        return copySents;
    }

    private static List<String> copyWithoutEmptyStrings(List<String> tokens){
        List<String> newTokens = new ArrayList<>();
        for(String tok: tokens){
            if(tok.trim().length() > 0){
                newTokens.add(tok.trim());
            }
        }
        return newTokens;
    }

    private static Tuple<List<List<String>>, Map<Integer,List<Tuple<Integer,Integer>>>> replaceMentions(Annotation document, List<List<String>> tokenSents) {

        Map<Integer, List<Tuple<Integer,Integer>>> affectedSpans = new HashMap<>();

        List<List<String>> replacedTokenSents = copySentences(tokenSents);
        for (CorefChain cc : document.get(CorefCoreAnnotations.CorefChainAnnotation.class).values()) {

            CorefChain.CorefMention representativeMention = cc.getRepresentativeMention();
            // note that all indexes and numbers are 1 based
            String headPhrase = representativeMention.mentionSpan;

            for (CorefChain.CorefMention mention : cc.getMentionsInTextualOrder()) {

                String mentionPhrase = mention.mentionSpan;
                // a lot of mentions are the same text
                if(mentionPhrase.equals(headPhrase)){
                    continue;
                }

                int sentNo = mention.sentNum - 1;

                int startIndex = mention.startIndex - 1;
                int endIndex = mention.endIndex - 2;

                List<String> sentence = replacedTokenSents.get(sentNo);
                if(!affectedSpans.containsKey(sentNo)){
                    affectedSpans.put(sentNo, new ArrayList<Tuple<Integer,Integer>>());
                }
                affectedSpans.get(sentNo).add(new Tuple<>(startIndex, endIndex));

                sentence.set(startIndex, "[[" + headPhrase.trim() + "]]");
                // wipe out the rest of the original mention
                for (int ix = startIndex + 1; ix <= endIndex; ix++) {
                    sentence.set(ix, "");
                }

            }
        }
        // removes empty strings
        replacedTokenSents = copySentences(replacedTokenSents);
        return new Tuple<List<List<String>>, Map<Integer,List<Tuple<Integer,Integer>>>>(replacedTokenSents, affectedSpans);
    }

    public static void main(String[] args) throws IOException {


        String folder = "/Users/simon.hughes/Google Drive/Phd/Data/CoralBleaching/Thesis_Dataset/CoReference/Training";
        List<String> filenames = findFiles(folder);

        // see https://stanfordnlp.github.io/CoreNLP/coref.html
        // NOTE - all of these other models are required to do co-ref resolution
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,depparse,coref");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        int fileNo = 0;
        Instant start = Instant.now();
        Instant globalStart = Instant.now();

        for(String fname: filenames){

            if(fileNo % 10 == 0){
                System.out.print(new Integer(fileNo));
                Instant end = Instant.now();
                Duration dur = Duration.between(start, end);
                System.out.print(" : ");
                System.out.print(dur.getSeconds()/10);
                System.out.println(" secs per doc");
                start = Instant.now();
            }

            String text = readFile(fname);
            Annotation document = new Annotation( text);
            pipeline.annotate(document);

            List<List<String>> tokenSents = getSentences(document);
            Tuple<List<List<String>>, Map<Integer,List<Tuple<Integer,Integer>>>> mentionsSpanPair = replaceMentions(document, tokenSents);

            List<List<String>> replacedTokenSents = mentionsSpanPair.x;
            Map<Integer,List<Tuple<Integer,Integer>>> affectedSpans = mentionsSpanPair.y;

            List<String> lines = new ArrayList<>();
            for(int sentIx = 0; sentIx < tokenSents.size(); sentIx++){

                String originalSentTokens = String.join(" ", tokenSents.get(sentIx));
                String replacedSentTokens = String.join("", replacedTokenSents.get(sentIx));

                String line = originalSentTokens + DELIM + replacedSentTokens + DELIM;
                if(affectedSpans.containsKey(sentIx)){
                    List<String> spanSection = new ArrayList<>();
                    List<Tuple<Integer,Integer>> spans = affectedSpans.get(sentIx);
                    for(Tuple<Integer,Integer> span: spans){
                        spanSection.add(span.x.toString() + "->"  + span.y.toString());
                    }
                    line += String.join(",", spanSection);
                }
                lines.add(line);
            }

            String outputFileName = fname + ".coref";
            Path file = Paths.get(outputFileName);
            Files.write(file, lines, Charset.forName("UTF-8"));

            fileNo++;
        }
        System.out.println();
        System.out.println("Done");
        Instant globalEnd = Instant.now();
        Duration dur = Duration.between(globalStart, globalEnd);
        System.out.print("Took: ");
        System.out.println(dur);
    }

    private static List<List<String>> getSentences(Annotation document) {
        List<List<String>> tokenSents = new ArrayList<List<String>>();

        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
        for(CoreMap sent: sentences){

            List<String> tokenSent = new ArrayList<>();
            for (CoreLabel token: sent.get(CoreAnnotations.TokensAnnotation.class)) {
                // this is the text of the token
                String word = token.get(CoreAnnotations.TextAnnotation.class);
                tokenSent.add(word);
            }
            tokenSents.add(tokenSent);
        }
        return tokenSents;
    }
}
