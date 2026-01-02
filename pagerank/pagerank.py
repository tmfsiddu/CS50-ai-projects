import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    N=len(corpus)
    probabilities={}
    #if pafe has no outer links
    if len(corpus[page]) == 0:
        for p in corpus:
            probabilities[p]=1/N
        return probabilities
    #base probability for all pages
    for p in corpus:
        probabilities[p]=(1-damping_factor)/N
    #adding extra probability for linked pages
    links=corpus[page]
    num_links=len(links)
    for linked_page in links:
        probabilities[linked_page]+=damping_factor/num_links
    return probabilities

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank={}
    for page in corpus:
        pagerank[page]=0
    #choose first page randomly
    current_page=random.choice(list(corpus.keys()))
    pagerank[current_page]+=1
    #generating remaining samples
    for i in range(n-1):
        distribution=transition_model(corpus,current_page,damping_factor)
        pages=list(distribution.keys())
        weights=list(distribution.values())
        current_page=random.choices(pages,weights=weights)[0]
        pagerank[current_page]+=1
    #convert count into probabilities
    for page in pagerank:
        pagerank[page]/=n
    return pagerank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N=len(corpus)
    pagerank={}
    for page in corpus:
        pagerank[page]=1/N
    while True:
        new_pagerank={}
        for page in corpus:
            new_pagerank[page]=(1-damping_factor)/N
            #adding incoming links
            for possible_page in corpus:
                if len(corpus[possible_page])==0:
                    new_pagerank[page]+=(
                        damping_factor*pagerank[possible_page]/N
                    )
                elif page in corpus[possible_page]:
                    new_pagerank[page]+=(
                        damping_factor*pagerank[possible_page]/len(corpus[possible_page])
                    )
        #difference checking
        difference=max(
            abs(new_pagerank[page]-pagerank[page])
            for page in pagerank
        )
        if difference<0.001:
            break
        pagerank=new_pagerank
    return pagerank




if __name__ == "__main__":
    main()
